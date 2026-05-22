"""
Stage 5 — Streamlit Dashboard.

Responsibility: Read-only browser interface. Loads from data/*.json only.
No database, no writes. All charts use Plotly. Uses st.cache_data on all
file loads.

Five views:
  1. Overview        — headline bias % per model, breakdown by category
  2. Probe explorer  — filter by category / model / label
  3. Probe detail    — single completion with SHAP bar chart
  4. Model comparison — side-by-side male vs female, GPT-4o vs Claude
  5. Export          — download filtered results as CSV

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Gender Bias Audit Dashboard", layout="wide")
st.title("Gender Bias Audit Dashboard")
st.info("Dashboard not yet implemented.")
