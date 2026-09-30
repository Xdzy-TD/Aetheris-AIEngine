"""
LLM-Powered Planner — replaces deterministic fallback routing with Ollama-driven tool-chain planning.

Uses the :class:`OllamaClient` to send the user's query plus the full set of
tool schemas to a local LLM, which returns a structured JSON plan specifying
which specialist tools to invoke and in what order.

This is the module that makes the controller *genuinely agentic*: the LLM
inspects the query semantics, image metadata, and tool capabilities to
compose a dynamic tool chain — not a fixed pipeline.

Fallback behaviour:
    If Ollama is unreachable or the LLM output is unparseable, every method
    returns ``None`` so the caller (:class:`AgenticPlanner`) can fall back
    to the existing deterministic fallback routing with zero disruption.
"""

from __future__ import annotations

import json
from typing import Any, cast

import structlog

from controller.ollama_client import OllamaClient, OllamaUnavailableError
from controller.schemas import ExecutionPlan, Query, ToolCall

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are the Aetheris Agentic Controller — a tool-chain planner for satellite \
remote-sensing analysis. Your job is to inspect the user's natural language \
query and the available specialist tools, then compose an ORDERED sequence of \
tool invocations that will answer the query.

Rules:
1. Return ONLY valid JSON matching the schema below. No explanation text.
2. Each step must reference a tool from the AVAILABLE TOOLS list.
3. Include a brief rationale for each tool choice.
4. Order matters: dependent steps must come after their dependencies.
5. Always include at least one step. The ``vqa_caption`` tool should almost \
   always be the final step since it synthesizes the answer.
6. If exactly one image is SAR and no optical/multispectral image is paired with it, \
   prepend ``sar_bridge`` to handle domain bridging.
7. If one optical/multispectral image and one SAR image of the same area are both \
   provided (a cross-modal pair), use ``optical_sar_fusion`` instead of ``sar_bridge`` \
   or ``change_detection`` — it extracts the complementary information from the pair.
8. If two images of the *same* modality are provided (a bi-temporal pair) and the query \
   asks about change/difference between them, include ``change_detection``. If a SAR pair \
   of the same two acquisitions is *also* provided, or both images carry acquisition dates, \
   prefer ``fusion_change_detection`` instead — it fuses optical, SAR and temporal evidence \
   through a learned model rather than thresholding one signal alone.
9. If the query asks to locate/segment/mask a feature, include ``grounding``.
10. If the query could benefit from geographic context (elevation, land cover, OSM), \
    include ``geo_rag``.
11. Use ``depends_on`` to declare data dependencies between steps.

Output JSON schema:
{
  "steps": [
    {
      "tool_name": "<tool_registry_key>",
      "arguments": { "<param>": "<value>", ... },
      "rationale": "<why this tool is needed>",
      "depends_on": ["<tool_name>", ...]
    }
  ],
  "strategy_note": "<one-sentence summary of the overall plan>"
}
"""

_SYNTHESIZER_SYSTEM_PROMPT = """\
You are the Aetheris answer synthesizer. Given a user's question about \
satellite imagery and the outputs from specialist analysis tools, \
compose a clear, structured report.

Format your response with these sections:

## Summary
One or two sentence answer to the user's question.

## Findings
- Per-specialist bullet points with specific numbers and measurements.
- Include land-cover percentages, change detection results, areas, etc.

## Confidence Assessment
- Report the confidence level from the analysis.
- Note any caveats or limitations.

## Methodology
- Which analysis tools were used and why.
- Note whether classical CV baselines or ML models were employed.

Rules:
1. Be concise but thorough — include specific numbers from specialist outputs.
2. If any specialist flagged low confidence, mention the uncertainty.
3. Do NOT invent information not present in the specialist outputs.
4. Always use the section headers above.
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class LLMPlanner:
    """LLM-powered tool-chain planner using Ollama.

    This class handles two responsibilities:
    1. **Planning**: composing a sequence of tool calls from the query.
    2. **Synthesis**: combining specialist outputs into a final answer.
    """

    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self._available = True  # optimistic; set False on first failure

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def plan(
        self,
        query: Query,
        tool_schemas: list[dict[str, Any]],
    ) -> ExecutionPlan | None:
        """Compose an execution plan using the LLM.

        Args:
            query:        The user's parsed query.
            tool_schemas: Tool schemas from ``ToolRegistry.get_schemas_for_planner()``.

        Returns:
            An :class:`ExecutionPlan` if successful, or ``None`` if Ollama
            is unreachable or the output is unparseable (signaling the
            caller to fall back to deterministic fallback routing).
        """
        if not self._available:
            # Quick check — re-probe periodically
            if not await self.client.health_check():
                return None
            self._available = True

        # Build the user message with context
        user_message = self._build_plan_prompt(query, tool_schemas)

        try:
            result = await self.client.chat_json(
                messages=[{"role": "user", "content": user_message}],
                system=_PLANNER_SYSTEM_PROMPT,
                temperature=0.1,
            )

            plan = self._parse_plan(query.query_id, result)
            if plan:
                logger.info(
                    "llm_plan_created",
                    query_id=query.query_id,
                    steps=[s.tool_name for s in plan.steps],
                    model=self.client.model,
                )
            return plan

        except OllamaUnavailableError as exc:
            logger.warning("llm_planner_unavailable", error=str(exc))
            self._available = False
            return None

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "llm_plan_parse_failed",
                error=str(exc),
                query_id=query.query_id,
            )
            return None

        except Exception as exc:
            logger.error(
                "llm_plan_unexpected_error",
                error=str(exc),
                query_id=query.query_id,
            )
            return None

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        query: Query,
        specialist_outputs: list[dict[str, Any]],
    ) -> str | None:
        """Combine specialist outputs into a natural language answer.

        Args:
            query:              The original user query.
            specialist_outputs: List of dicts from each specialist's ``run()``.

        Returns:
            A synthesized answer string, or ``None`` to fall back to
            the default synthesizer.
        """
        if not self._available:
            if not await self.client.health_check():
                return None
            self._available = True

        user_message = self._build_synthesis_prompt(query, specialist_outputs)

        try:
            response = await self.client.chat(
                messages=[{"role": "user", "content": user_message}],
                system=_SYNTHESIZER_SYSTEM_PROMPT,
                format=None,  # free-form text, not JSON
                temperature=0.3,
            )

            answer = cast(str, response.get("message", {}).get("content", "")).strip()
            if answer:
                logger.info(
                    "llm_synthesis_complete",
                    query_id=query.query_id,
                    answer_length=len(answer),
                )
                return answer
            return None

        except OllamaUnavailableError:
            self._available = False
            return None

        except Exception as exc:
            logger.warning("llm_synthesis_failed", error=str(exc))
            return None

    async def generate_report(
        self,
        query: Query,
        specialist_output: dict[str, Any],
    ) -> str | None:
        """Generate a structured text report even for a single specialist output.

        Unlike :meth:`synthesize` (which takes a list), this is designed for
        the common case of a single-tool query that still needs a readable report.
        """
        return await self.synthesize(query, [specialist_output])

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """Check whether the LLM planner can be used right now."""
        self._available = await self.client.health_check()
        return self._available

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plan_prompt(
        query: Query,
        tool_schemas: list[dict[str, Any]],
    ) -> str:
        """Build the user-role prompt for planning."""
        # Describe the images
        image_context = ""
        if query.images:
            parts = []
            for i, img in enumerate(query.images, 1):
                desc = f"Image {i}: {img.filename}, modality={img.modality.value}"
                if img.sensor:
                    desc += f", sensor={img.sensor}"
                if img.resolution_m:
                    desc += f", resolution={img.resolution_m}m"
                if img.bands:
                    desc += f", bands={img.bands}"
                if img.bbox:
                    desc += f", bbox={img.bbox}"
                parts.append(desc)
            image_context = "\n".join(parts)
        else:
            image_context = "No images uploaded (text-only query)."

        # Format tool schemas
        tools_json = json.dumps(tool_schemas, indent=2)

        return (
            f"USER QUERY: {query.text}\n\n"
            f"INPUT IMAGES:\n{image_context}\n\n"
            f"AVAILABLE TOOLS:\n{tools_json}\n\n"
            f"Compose the optimal tool-chain plan as JSON."
        )

    @staticmethod
    def _build_synthesis_prompt(
        query: Query,
        specialist_outputs: list[dict[str, Any]],
    ) -> str:
        """Build the user-role prompt for answer synthesis."""
        outputs_formatted = []
        for out in specialist_outputs:
            tool_name = out.get("tool_name", "unknown")
            result = out.get("result", out)
            outputs_formatted.append(
                f"--- {tool_name} ---\n{json.dumps(result, indent=2, default=str)}"
            )

        outputs_text = "\n\n".join(outputs_formatted)

        image_context = ""
        if query.images:
            parts = []
            for img in query.images:
                desc = f"{img.filename} ({img.modality.value})"
                if img.sensor:
                    desc += f", sensor={img.sensor}"
                if img.resolution_m:
                    desc += f", {img.resolution_m}m resolution"
                parts.append(desc)
            image_context = "Images: " + "; ".join(parts)

        return (
            f"USER QUESTION: {query.text}\n\n"
            f"{image_context}\n\n"
            f"SPECIALIST ANALYSIS OUTPUTS:\n{outputs_text}\n\n"
            f"Compose a clear, accurate answer to the user's question."
        )

    # ------------------------------------------------------------------
    # Plan parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(
        query_id: str,
        raw: dict[str, Any],
    ) -> ExecutionPlan | None:
        """Parse the LLM's JSON output into an ExecutionPlan.

        Returns None if the structure is invalid.
        """
        steps_raw = raw.get("steps", [])
        if not isinstance(steps_raw, list) or not steps_raw:
            logger.warning("llm_plan_empty_steps", raw_keys=list(raw.keys()))
            return None

        steps: list[ToolCall] = []
        for step_data in steps_raw:
            if not isinstance(step_data, dict):
                continue
            tool_name = step_data.get("tool_name", "")
            if not tool_name:
                continue

            steps.append(ToolCall(
                tool_name=tool_name,
                arguments=step_data.get("arguments", {}),
                rationale=step_data.get("rationale", ""),
                depends_on=step_data.get("depends_on", []),
            ))

        if not steps:
            return None

        strategy_note = raw.get("strategy_note", "LLM-planned tool chain")

        return ExecutionPlan(
            query_id=query_id,
            steps=steps,
            strategy_note=strategy_note,
            planning_method="llm",
        )
