"""Unit tests for Stage 1 probe bank validator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.stage1_probe_bank import (
    ProbeEntry,
    load_probe_bank,
    log_summary,
    main,
    validate_probe_bank,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _make_probe(**overrides: str) -> ProbeEntry:
    defaults: dict[str, str] = {
        "probe_id": "wb_001",
        "category": "professional_role",
        "role": "engineer",
        "source": "winobias",
        "male_prompt": "A male engineer in a meeting.",
        "female_prompt": "A female engineer in a meeting.",
    }
    defaults.update(overrides)
    return ProbeEntry(**defaults)


def _make_probes(
    n: int,
    *,
    source: str = "winobias",
    category: str = "professional_role",
) -> list[ProbeEntry]:
    prefix = "wb" if source == "winobias" else "orig"
    return [
        _make_probe(
            probe_id=f"{prefix}_{i:03d}",
            source=source,
            category=category,
        )
        for i in range(1, n + 1)
    ]


class TestLoadProbeBank:
    def test_loads_valid_fixture(self) -> None:
        probes = load_probe_bank(_FIXTURES_DIR / "probe_bank.json")
        assert len(probes) > 0
        required = {
            "probe_id",
            "category",
            "role",
            "source",
            "male_prompt",
            "female_prompt",
        }
        assert set(probes[0].model_dump().keys()) == required

    def test_all_fields_populated(self) -> None:
        probes = load_probe_bank(_FIXTURES_DIR / "probe_bank.json")
        for p in probes:
            assert p.probe_id and p.category and p.role
            assert p.source and p.male_prompt and p.female_prompt

    def test_male_and_female_prompts_differ(self) -> None:
        probes = load_probe_bank(_FIXTURES_DIR / "probe_bank.json")
        for p in probes:
            assert p.male_prompt != p.female_prompt

    def test_raises_on_missing_field(self, tmp_path: Path) -> None:
        bad = [{"probe_id": "wb_001", "category": "professional_role"}]
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="schema validation"):
            load_probe_bank(f)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_probe_bank(f)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_probe_bank(tmp_path / "nonexistent.json")


class TestValidateProbeBank:
    def test_passes_with_valid_150_probes(self) -> None:
        validate_probe_bank(_make_probes(150))

    def test_raises_on_count_too_low(self) -> None:
        with pytest.raises(ValueError, match="Expected 150"):
            validate_probe_bank(_make_probes(149))

    def test_raises_on_count_too_high(self) -> None:
        with pytest.raises(ValueError, match="Expected 150"):
            validate_probe_bank(_make_probes(151))

    def test_raises_on_duplicate_ids(self) -> None:
        probes = _make_probes(150)
        probes[5] = _make_probe(probe_id="wb_001")
        with pytest.raises(ValueError, match="Duplicate"):
            validate_probe_bank(probes)

    def test_raises_on_invalid_source(self) -> None:
        probes = _make_probes(150)
        probes[0] = _make_probe(probe_id="wb_001", source="stereoset")
        with pytest.raises(ValueError, match="Invalid source"):
            validate_probe_bank(probes)

    def test_raises_on_invalid_category(self) -> None:
        probes = _make_probes(150)
        probes[0] = _make_probe(probe_id="wb_001", category="unknown")
        with pytest.raises(ValueError, match="Invalid category"):
            validate_probe_bank(probes)

    def test_accepts_all_valid_sources(self) -> None:
        probes = [
            _make_probe(probe_id=f"wb_{i:03d}", source="winobias") for i in range(1, 76)
        ] + [
            _make_probe(probe_id=f"orig_{i:03d}", source="original")
            for i in range(1, 76)
        ]
        validate_probe_bank(probes)

    def test_accepts_all_valid_categories(self) -> None:
        cats = [
            "professional_role",
            "personality_trait",
            "ambiguous_scenario",
            "coreference_ambiguity",
        ]
        probes = [
            _make_probe(probe_id=f"wb_{i:03d}", category=cats[i % 4])
            for i in range(1, 151)
        ]
        validate_probe_bank(probes)


class TestLogSummary:
    def test_calls_logger_info(self) -> None:
        with patch("src.stage1_probe_bank.logger") as mock_logger:
            log_summary(_make_probes(5))
        assert mock_logger.info.called

    def test_does_not_raise(self) -> None:
        log_summary(_make_probes(5))


class TestMain:
    def test_calls_load_validate_and_log(self, tmp_path: Path) -> None:
        config = {"paths": {"probe_bank": str(tmp_path / "probe_bank.json")}}
        probes = _make_probes(150)
        with (
            patch("src.stage1_probe_bank._load_config", return_value=config),
            patch(
                "src.stage1_probe_bank.load_probe_bank", return_value=probes
            ) as mock_load,
            patch("src.stage1_probe_bank.validate_probe_bank") as mock_validate,
            patch("src.stage1_probe_bank.log_summary") as mock_log,
        ):
            main()
        mock_load.assert_called_once_with(Path(config["paths"]["probe_bank"]))
        mock_validate.assert_called_once_with(probes)
        mock_log.assert_called_once_with(probes)

    def test_propagates_validation_error(self, tmp_path: Path) -> None:
        config = {"paths": {"probe_bank": str(tmp_path / "probe_bank.json")}}
        with (
            patch("src.stage1_probe_bank._load_config", return_value=config),
            patch("src.stage1_probe_bank.load_probe_bank", return_value=[]),
            patch(
                "src.stage1_probe_bank.validate_probe_bank",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(ValueError, match="bad"),
        ):
            main()

    def test_does_not_write_file(self, tmp_path: Path) -> None:
        output = tmp_path / "probe_bank.json"
        config = {"paths": {"probe_bank": str(output)}}
        probes = _make_probes(150)
        with (
            patch("src.stage1_probe_bank._load_config", return_value=config),
            patch("src.stage1_probe_bank.load_probe_bank", return_value=probes),
            patch("src.stage1_probe_bank.validate_probe_bank"),
            patch("src.stage1_probe_bank.log_summary"),
        ):
            main()
        assert not output.exists()
