# Gender Bias Auditing Dashboard

A five-stage automated pipeline that detects implicit gender bias in LLM outputs using a fine-tuned RoBERTa classifier with SHAP token-level explainability. Results surface in a Streamlit dashboard comparing bias scores across models side by side.

**Louise Slattery — MSc Computer Science**

---

## Dataset design

| Stage | Dataset | Purpose |
|---|---|---|
| Stage 1  probe bank | WinoBias + original probes | Restructured into conversational LLM prompts |
| Stage 2  LLM query runner | GPT-4o and claude-opus-4-5 | Queries both models with all probes, collects LLM text completions |
| Stage 3  classifier fine-tuning | StereoSet data used to fine-tune classifier | Fine-tunes pretrained RoBERTa base model with a classification head added, for binary bias detection |
| Stage 4  SHAP explainability | Applies SHAP to the fine-tuned classifier to produce token-level attribution scores |


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
# Run all stages in sequence
python run_pipeline.py

# Or run stages individually:
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
