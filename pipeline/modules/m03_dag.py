"""
M03 — Execution DAG Orchestrator.

Represents M01–M10 as a directed acyclic graph with real dependency
tracking, cycle detection, topological ordering, and state management.
Delegates to the old AETHERIS planner for actual analysis execution.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from enum import StrEnum
from typing import Any, Callable

from pipeline.schemas.contracts import (
    ModuleID,
    ModuleResult,
    ResultStatus,
)


class NodeState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class DAGNode:
    """A single node in the execution DAG."""

    def __init__(
        self,
        node_id: str,
        module_id: ModuleID,
        execute_fn: Callable[..., ModuleResult],
        depends_on: list[str] | None = None,
        optional: bool = False,
    ) -> None:
        self.node_id = node_id
        self.module_id = module_id
        self.execute_fn = execute_fn
        self.depends_on = depends_on or []
        self.optional = optional
        self.state = NodeState.PENDING
        self.result: ModuleResult | None = None
        self.execution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "module": self.module_id.value,
            "state": self.state.value,
            "depends_on": self.depends_on,
            "optional": self.optional,
            "execution_time_s": self.execution_time_s,
            "has_result": self.result is not None,
        }


class ExecutionDAG:
    """Directed acyclic graph for M01–M10 pipeline execution."""

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}
        self._execution_order: list[str] | None = None

    def add_node(self, node: DAGNode) -> None:
        self._nodes[node.node_id] = node
        self._execution_order = None  # invalidate cache

    def validate(self) -> ModuleResult:
        """Validate the DAG structure — check for cycles and missing deps."""
        t0 = time.perf_counter()
        errors = []

        # Check for missing dependencies
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    errors.append(f"Node '{node.node_id}' depends on unknown node '{dep}'")

        # Check for cycles using DFS
        if not errors:
            cycle = self._detect_cycle()
            if cycle:
                errors.append(f"Cycle detected: {' -> '.join(cycle)}")

        if errors:
            return ModuleResult.failure(
                ModuleID.M03,
                "DAG_VALIDATION_FAILED",
                "; ".join(errors),
            )

        # Compute execution order
        self._execution_order = self._topological_sort()
        return ModuleResult.success(
            ModuleID.M03,
            {
                "valid": True,
                "node_count": len(self._nodes),
                "execution_order": self._execution_order,
            },
            exec_time=round(time.perf_counter() - t0, 4),
        )

    def _detect_cycle(self) -> list[str] | None:
        """Detect cycles using DFS with coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}
        parent: dict[str, str | None] = {nid: None for nid in self._nodes}

        def dfs(nid: str) -> list[str] | None:
            color[nid] = GRAY
            for dep in self._nodes[nid].depends_on:
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    # Found cycle — walk parent pointers back from nid to
                    # dep to get the *full* cycle, not just the closing
                    # edge. The old code returned only [dep, nid], which
                    # for any cycle longer than 2 nodes silently hid the
                    # nodes in between from the reported error message.
                    cycle = [nid]
                    cursor = nid
                    while cursor != dep:
                        cursor = parent[cursor]
                        cycle.append(cursor)
                    cycle.reverse()
                    cycle.append(dep)
                    return cycle
                if color[dep] == WHITE:
                    parent[dep] = nid
                    result = dfs(dep)
                    if result:
                        return result
            color[nid] = BLACK
            return None

        for nid in self._nodes:
            if color[nid] == WHITE:
                result = dfs(nid)
                if result:
                    return result
        return None

    def _topological_sort(self) -> list[str]:
        """Topological sort using Kahn's algorithm."""
        in_degree: dict[str, int] = defaultdict(int)
        for nid in self._nodes:
            in_degree[nid] = 0
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep in self._nodes:
                    in_degree[node.node_id] += 1

        # Reverse: in_degree counts incoming edges
        # Actually we need to count how many deps each node has
        in_degree = {}
        adj: dict[str, list[str]] = defaultdict(list)
        for node in self._nodes.values():
            in_degree[node.node_id] = len([d for d in node.depends_on if d in self._nodes])
            for dep in node.depends_on:
                if dep in self._nodes:
                    adj[dep].append(node.node_id)

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for successor in adj[nid]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return order

    def execute(self, context: dict[str, Any] | None = None) -> ModuleResult:
        """Execute all nodes in topological order."""
        t0 = time.perf_counter()
        ctx = context or {}

        # Validate first
        validation = self.validate()
        if validation.status != ResultStatus.SUCCESS:
            return validation

        assert self._execution_order is not None
        results: dict[str, ModuleResult] = {}

        for node_id in self._execution_order:
            node = self._nodes[node_id]

            # Check if dependencies succeeded
            deps_ok = True
            for dep in node.depends_on:
                dep_node = self._nodes.get(dep)
                if dep_node and dep_node.state == NodeState.FAILED and not dep_node.optional:
                    deps_ok = False
                    break

            if not deps_ok:
                node.state = NodeState.SKIPPED
                node.result = ModuleResult.skipped(node.module_id, "dependency failed")
                results[node_id] = node.result
                continue

            # Execute
            node.state = NodeState.RUNNING
            nt0 = time.perf_counter()
            try:
                node.result = node.execute_fn(ctx)
                node.execution_time_s = round(time.perf_counter() - nt0, 4)
                node.result.execution_time_s = node.execution_time_s
                if node.result.status == ResultStatus.FAILURE and not node.optional:
                    node.state = NodeState.FAILED
                else:
                    node.state = NodeState.DONE
            except Exception as exc:
                node.state = NodeState.FAILED
                node.execution_time_s = round(time.perf_counter() - nt0, 4)
                node.result = ModuleResult.failure(
                    node.module_id, "EXECUTION_ERROR", str(exc),
                )
            results[node_id] = node.result

        elapsed = round(time.perf_counter() - t0, 4)
        return ModuleResult.success(
            ModuleID.M03,
            {
                "dag_execution": "complete",
                "node_results": {
                    nid: {
                        "state": self._nodes[nid].state.value,
                        "status": r.status.value if r else "unknown",
                    }
                    for nid, r in results.items()
                },
                "execution_order": self._execution_order,
            },
            exec_time=elapsed,
        )

    def get_state(self) -> dict[str, Any]:
        """Return the current DAG state."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "execution_order": self._execution_order,
            "total_nodes": len(self._nodes),
            "completed": sum(1 for n in self._nodes.values() if n.state == NodeState.DONE),
            "failed": sum(1 for n in self._nodes.values() if n.state == NodeState.FAILED),
            "pending": sum(1 for n in self._nodes.values() if n.state == NodeState.PENDING),
            "running": sum(1 for n in self._nodes.values() if n.state == NodeState.RUNNING),
            "skipped": sum(1 for n in self._nodes.values() if n.state == NodeState.SKIPPED),
        }

    def reset(self) -> None:
        """Reset all nodes to pending."""
        for node in self._nodes.values():
            node.state = NodeState.PENDING
            node.result = None
            node.execution_time_s = 0.0
