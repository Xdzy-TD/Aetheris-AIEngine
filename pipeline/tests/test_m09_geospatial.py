"""
Tests for M17 — Geospatial test suite (covers M09 CRS/spatial validation).

Mocks ImageryAdapter rather than requiring real rasterio/GDAL rasters, so
this passes on the bare harness install same as everything else (see
harness/README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.modules import m09_geospatial as m09
from pipeline.modules.m09_geospatial import check_crs_compatibility, get_spatial_summary, validate_crs


class TestValidateCRS:
    def test_rasterio_unavailable(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "check_rasterio_available", lambda: False)
        result = validate_crs("x.tif")
        assert result.status.value == "success"
        assert result.data["crs_valid"] is None
        assert result.data["rasterio_available"] is False

    def test_probe_error(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "check_rasterio_available", lambda: True)
        monkeypatch.setattr(m09.ImageryAdapter, "probe_metadata", lambda path: {"error": "boom"})
        result = validate_crs("x.tif")
        assert result.status.value == "failure"
        assert result.errors[0].code == "CRS_PROBE_ERROR"

    def test_not_georeferenced(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "check_rasterio_available", lambda: True)
        monkeypatch.setattr(m09.ImageryAdapter, "probe_metadata", lambda path: {"georeferenced": False})
        result = validate_crs("x.tif")
        assert result.data["crs_valid"] is False
        assert result.data["georeferenced"] is False

    def test_known_crs_no_warnings(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "check_rasterio_available", lambda: True)
        monkeypatch.setattr(m09.ImageryAdapter, "probe_metadata", lambda path: {
            "georeferenced": True, "crs": "EPSG:4326", "resolution_m": 10.0,
        })
        result = validate_crs("x.tif")
        assert result.data["crs_valid"] is True
        assert result.data["crs_description"] == "WGS 84 (Geographic)"
        assert result.warnings == []

    def test_unknown_crs_warns(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "check_rasterio_available", lambda: True)
        monkeypatch.setattr(m09.ImageryAdapter, "probe_metadata", lambda path: {
            "georeferenced": True, "crs": "EPSG:9999", "resolution_m": 10.0,
        })
        result = validate_crs("x.tif")
        assert any("not in common" in w for w in result.warnings)

    def test_resolution_out_of_bounds_warns(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "check_rasterio_available", lambda: True)
        monkeypatch.setattr(m09.ImageryAdapter, "probe_metadata", lambda path: {
            "georeferenced": True, "crs": "EPSG:4326", "resolution_m": 5000.0,
        })
        result = validate_crs("x.tif")
        assert any("unusually coarse" in w for w in result.warnings)


class TestCRSCompatibility:
    def test_both_none(self):
        assert check_crs_compatibility(None, None).data["compatible"] is True

    def test_one_none_fails(self):
        result = check_crs_compatibility("EPSG:4326", None)
        assert result.status.value == "failure"
        assert result.errors[0].code == "CRS_MISMATCH"

    def test_matching(self):
        result = check_crs_compatibility("EPSG:4326", "EPSG:4326")
        assert result.data["compatible"] is True

    def test_differing_needs_reprojection(self):
        result = check_crs_compatibility("EPSG:4326", "EPSG:32643")
        assert result.data["compatible"] is False
        assert result.data["reprojection_needed"] is True
        assert result.warnings


class TestSpatialSummary:
    def test_shape(self, monkeypatch):
        monkeypatch.setattr(m09.ImageryAdapter, "probe_metadata", lambda path: {
            "georeferenced": True, "crs": "EPSG:4326", "band_count": 3,
        })
        summary = get_spatial_summary("x.tif")
        assert summary["path"] == "x.tif"
        assert summary["crs"] == "EPSG:4326"
        assert summary["band_count"] == 3
