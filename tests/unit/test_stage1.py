"""Unit tests for Stage 1 probe bank builder."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.stage1_probe_bank import ProbeEntry, build_probe_bank, main, write_probe_bank

_EXPECTED_TOTAL = 150
_EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "professional_role": 50,
    "personality_trait": 40,
    "ambiguous_scenario": 40,
    "coreference_ambiguity": 20,
}
_VALID_CATEGORIES = frozenset(_EXPECTED_CATEGORY_COUNTS)
_VALID_SOURCES = frozenset({"winobias", "original"})
_ID_PATTERN = re.compile(r"^(wb|orig)_\d{3}$")


class TestBuildProbeBank:
    def test_returns_correct_total(self) -> None:
        assert len(build_probe_bank()) == _EXPECTED_TOTAL

    def test_all_entries_are_probe_entries(self) -> None:
        assert all(isinstance(p, ProbeEntry) for p in build_probe_bank())

    def test_no_duplicate_ids(self) -> None:
        ids = [p.probe_id for p in build_probe_bank()]
        assert len(ids) == len(set(ids))

    def test_category_counts(self) -> None:
        counts: dict[str, int] = {}
        for p in build_probe_bank():
            counts[p.category] = counts.get(p.category, 0) + 1
        assert counts == _EXPECTED_CATEGORY_COUNTS

    def test_winobias_probes_use_wb_prefix(self) -> None:
        for p in build_probe_bank():
            if p.source == "winobias":
                assert p.probe_id.startswith("wb_")

    def test_original_probes_use_orig_prefix(self) -> None:
        for p in build_probe_bank():
            if p.source == "original":
                assert p.probe_id.startswith("orig_")

    def test_probe_id_format(self) -> None:
        for p in build_probe_bank():
            assert _ID_PATTERN.match(p.probe_id), f"Invalid probe_id: {p.probe_id}"

    def test_no_ss_prefix_in_ids(self) -> None:
        for p in build_probe_bank():
            assert not p.probe_id.startswith("ss_")

    def test_all_required_fields_present(self) -> None:
        required = {
            "probe_id",
            "category",
            "role",
            "source",
            "male_prompt",
            "female_prompt",
        }
        for p in build_probe_bank():
            assert set(p.model_dump().keys()) == required

    def test_prompts_are_non_empty(self) -> None:
        for p in build_probe_bank():
            assert p.male_prompt.strip()
            assert p.female_prompt.strip()

    def test_male_and_female_prompts_differ(self) -> None:
        for p in build_probe_bank():
            assert p.male_prompt != p.female_prompt

    def test_sources_are_valid(self) -> None:
        for p in build_probe_bank():
            assert p.source in _VALID_SOURCES

    def test_categories_are_valid(self) -> None:
        for p in build_probe_bank():
            assert p.category in _VALID_CATEGORIES

    def test_coreference_ambiguity_all_winobias(self) -> None:
        coref = [p for p in build_probe_bank() if p.category == "coreference_ambiguity"]
        assert all(p.source == "winobias" for p in coref)

    def test_personality_trait_all_original(self) -> None:
        traits = [p for p in build_probe_bank() if p.category == "personality_trait"]
        assert all(p.source == "original" for p in traits)

    def test_ambiguous_scenario_all_original(self) -> None:
        scenarios = [
            p for p in build_probe_bank() if p.category == "ambiguous_scenario"
        ]
        assert all(p.source == "original" for p in scenarios)

    def test_professional_role_has_both_sources(self) -> None:
        prof = [p for p in build_probe_bank() if p.category == "professional_role"]
        sources = {p.source for p in prof}
        assert "winobias" in sources
        assert "original" in sources

    def test_wb_ids_are_sequential(self) -> None:
        wb_ids = [
            p.probe_id for p in build_probe_bank() if p.probe_id.startswith("wb_")
        ]
        numbers = [int(pid.split("_")[1]) for pid in wb_ids]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_orig_ids_are_sequential(self) -> None:
        orig_ids = [
            p.probe_id for p in build_probe_bank() if p.probe_id.startswith("orig_")
        ]
        numbers = [int(pid.split("_")[1]) for pid in orig_ids]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_roles_are_non_empty(self) -> None:
        for p in build_probe_bank():
            assert p.role.strip()


class TestWriteProbeBank:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        probes = build_probe_bank()
        output = tmp_path / "probe_bank.json"
        write_probe_bank(probes, output)
        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == _EXPECTED_TOTAL

    def test_output_schema(self, tmp_path: Path) -> None:
        output = tmp_path / "probe_bank.json"
        write_probe_bank(build_probe_bank()[:1], output)
        with open(output) as f:
            data = json.load(f)
        expected_keys = {
            "probe_id",
            "category",
            "role",
            "source",
            "male_prompt",
            "female_prompt",
        }
        assert set(data[0].keys()) == expected_keys

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = tmp_path / "data" / "nested" / "probe_bank.json"
        write_probe_bank(build_probe_bank()[:1], output)
        assert output.exists()

    def test_roundtrip_probe_id(self, tmp_path: Path) -> None:
        probes = build_probe_bank()
        output = tmp_path / "probe_bank.json"
        write_probe_bank(probes, output)
        with open(output) as f:
            data = json.load(f)
        written_ids = [entry["probe_id"] for entry in data]
        original_ids = [p.probe_id for p in probes]
        assert written_ids == original_ids


class TestMain:
    def test_main_writes_probe_bank(self, tmp_path: Path) -> None:
        config = {
            "probe_bank": {"target_pairs": 150},
            "paths": {"probe_bank": str(tmp_path / "probe_bank.json")},
        }
        with patch("src.stage1_probe_bank._load_config", return_value=config):
            main()
        output = Path(config["paths"]["probe_bank"])
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert len(data) == _EXPECTED_TOTAL

    def test_main_raises_if_target_not_met(self, tmp_path: Path) -> None:
        config = {
            "probe_bank": {"target_pairs": 9999},
            "paths": {"probe_bank": str(tmp_path / "probe_bank.json")},
        }
        with (
            patch("src.stage1_probe_bank._load_config", return_value=config),
            pytest.raises(ValueError, match="target"),
        ):
            main()

    def test_main_output_matches_schema(self, tmp_path: Path) -> None:
        config = {
            "probe_bank": {"target_pairs": 150},
            "paths": {"probe_bank": str(tmp_path / "probe_bank.json")},
        }
        with patch("src.stage1_probe_bank._load_config", return_value=config):
            main()
        with open(tmp_path / "probe_bank.json") as f:
            data = json.load(f)
        required = {
            "probe_id",
            "category",
            "role",
            "source",
            "male_prompt",
            "female_prompt",
        }
        for entry in data:
            assert set(entry.keys()) == required
