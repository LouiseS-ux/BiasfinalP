# Gender Bias Detector Dashboard
**Louise Slattery — MSc Computer Science**

A five-stage pipeline detecting gender bias in LLM outputs using the pretrained HEARTS ALBERT-v2 classifier with SHAP token-level explainability and divergence testing. The dashboard is a separate, read only Streamlit app that reads the pipeline's existing static output files.



**View Dashboard here:** *[add Streamlit link once deployed]*

---

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1 — Probe bank | `src/stage1_probe_bank.py` | 250 gender-paired probes, 6 categories → `data/probe_bank.json` |
| 2 — LLM query | `src/stage2_query_llm.py` | 1,000 completions (GPT-4o + Claude) → `data/completions.json` |
| 3 — HEARTS classifier | `src/stage3_hearts.py` | Pretrained HEARTS ALBERT-v2, no fine-tuning → `data/hearts_predictions.json`, `data/hearts_summary.json` |
| 3b — Divergence | `src/stage3b_divergence.py` | Male vs. female label comparison → `data/divergence_results.json` |
| 4 — SHAP | `src/stage4_shap.py` | Token-level attribution → `data/hearts_shap_values.json` | data/hearts_shap_summary.json 
| 5 — Dashboard | `src/dashboard/app.py` | It reads directly from the files already in `data/` each time it loads; does not trigger or depend on running Stages 1–4 |

Note: `src/stage3_roberta_legacy.py` is retained for reference only — superseded by HEARTS, not part of the active pipeline.

---

## Quick start — view dashboard only, which is the intended route to show results.

No API keys needed; pre-generated data files are already committed.

```bash
git clone https://github.com/LouiseS-ux/BiasfinalP
cd BiasfinalP
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run src/dashboard/app.py
```

---

## Full setup — For reference here are instructions to rerun pipeline

**Warning:** rerunning the pipeline overwrites the data files the dashboard currently reads from. Stage 2's LLM completions are not guaranteed to be identical on a rerun, so classifier results, bias percentages, and divergence figures may differ from those currently shown on the dashboard and reported in the dissertation.

**Instead, please view the existing, documented results using the Quick start section above — no new pipeline run needed.**

------------------------------------------------------------------------------
Just for reference: To run all stages in sequence for full setup:
```bash
python run_pipeline.py
```

```bash
git clone https://github.com/LouiseS-ux/BiasfinalP
cd BiasfinalP
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# add OPENAI_API_KEY and ANTHROPIC_API_KEY to .env
pre-commit install
```

Run stages in order:
```bash
python src/stage1_probe_bank.py
python src/stage2_query_llm.py
python src/stage3_hearts.py
python src/stage3b_divergence.py
python src/stage4_shap.py
```

HEARTS ALBERT-v2 downloads automatically from HuggingFace on first run.
---

## Tests

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

---

## Configuration

All paths and settings are in `config.yaml`
—  active classifier uses (HEARTS) not legacy (RoBERTa).
