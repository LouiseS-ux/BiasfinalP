# Gender Bias Auditing Dashboard

A five-stage automated pipeline that detects implicit gender bias in LLM outputs using a fine-tuned RoBERTa classifier with SHAP token-level explainability. Results surface in a Streamlit dashboard comparing bias scores across models side by side.

**Louise Slattery — MSc Computer Science**

---

## Dataset design

| Stage | Dataset | Purpose |
|---|---|---|
| Stage 1 — probe bank | WinoBias + original probes | Restructured into conversational LLM prompts |
| Stage 3 — RoBERTa training | StereoSet only | Labelled training data for the classifier |

Datasets are kept separate to avoid data leakage — RoBERTa never trains on the same source material used to build the probes.

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/LouiseS-ux/BiasfinalP
cd BiasfinalP
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Add API keys
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

### 3. Install pre-commit hooks
```bash
pre-commit install
```

---

## Run the pipeline

```bash
python src/stage1_probe_bank.py
python src/stage2_query_llm.py
python src/stage3_classifier.py --mode train
python src/stage3_classifier.py --mode inference
python src/stage4_shap.py
streamlit run src/dashboard/app.py
```

---

## Run tests

```bash
pytest
pytest tests/unit/
pytest tests/integration/
```

---

## Model checkpoint

Not committed to Git due to file size. Reproduce by running Stage 3 train after Stage 2 completes.
