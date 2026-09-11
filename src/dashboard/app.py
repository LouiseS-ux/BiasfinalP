"""
Stage 5 — Streamlit Dashboard.

Read-only browser interface. Loads from data/*.json + hearts_divergence_results.txt
only. Uses st.cache_data on all file loads.

Two tabs:
  1. Overview            — bias % and gender divergence per model, by category
  2. SHAP explainability — flagged-token word cloud, probe selector with
                            highlighted male/female completions

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")


class Prediction(BaseModel):
    """One HEARTS classifier prediction for a (probe, gender, model) completion."""

    probe_id: str
    gender: str
    model: str
    category: str
    label: str
    confidence: float
    completion: str


class ShapRecord(BaseModel):
    """Token-level SHAP attribution for a single (probe, gender, model) completion."""

    probe_id: str
    gender: str
    model: str
    tokens: list[str]
    shap: list[float]


class Probe(BaseModel):
    """A gender-paired probe from the Stage 1 probe bank."""

    probe_id: str
    category: str
    role: str
    source: str
    male_prompt: str
    female_prompt: str
    category_index: int
    overall_index: int


class CategoryStats(BaseModel):
    """Aggregated bias stats for one probe category, for one model."""

    total: int
    biased_count: int
    bias_pct: float
    mean_confidence_biased: float


class ModelSummary(BaseModel):
    """Aggregated bias stats for one model across all categories."""

    total: int
    biased_count: int
    bias_pct: float
    mean_confidence_biased: float
    by_category: dict[str, CategoryStats]


class HeartsSummary(BaseModel):
    """Top-level Stage 3 (HEARTS) aggregate summary."""

    generated_at: str
    classifier: str
    total_completions: int
    total_biased: int
    overall_bias_pct: float
    by_model: dict[str, ModelSummary]


class DivergentPair(BaseModel):
    """A single gendered probe pair where the male/female labels diverged."""

    model: str
    probe_id: str
    male_label: str
    male_confidence: float
    female_label: str
    female_confidence: float
    male_text: str
    female_text: str


class DivergenceSummary(BaseModel):
    """Parsed contents of hearts_divergence_results.txt."""

    total_pairs: int
    divergent_count: int
    divergent_pct: float
    male_biased: int
    female_biased: int
    by_model: dict[str, int]
    pairs: list[DivergentPair]


@st.cache_data
def load_config() -> dict:
    """Load pipeline configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", CONFIG_PATH)
    return config


@st.cache_data
def load_predictions(path: str) -> list[Prediction]:
    """Load and validate hearts_predictions.json."""
    with open(path) as f:
        raw = json.load(f)
    predictions = [Prediction(**row) for row in raw]
    logger.info("Loaded %d predictions from %s", len(predictions), path)
    return predictions


@st.cache_data
def load_shap_values(path: str) -> list[ShapRecord]:
    """Load and validate hearts_shap_values.json."""
    with open(path) as f:
        raw = json.load(f)
    shap_records = [ShapRecord(**row) for row in raw]
    logger.info("Loaded %d SHAP records from %s", len(shap_records), path)
    return shap_records


@st.cache_data
def load_probe_bank(path: str) -> list[Probe]:
    """Load and validate probe_bank.json."""
    with open(path) as f:
        raw = json.load(f)
    probes = [Probe(**row) for row in raw]
    logger.info("Loaded %d probes from %s", len(probes), path)
    return probes


@st.cache_data
def load_summary(path: str) -> HeartsSummary:
    """Load and validate hearts_summary.json."""
    with open(path) as f:
        raw = json.load(f)
    summary = HeartsSummary(**raw)
    logger.info("Loaded HEARTS summary from %s", path)
    return summary


_PAIR_LINE = re.compile(
    r"^(?P<model>[\w.\-]+) \| (?P<probe_id>\S+) \| "
    r"male: (?P<male_label>\w+) \((?P<male_conf>[\d.]+)\) \| "
    r"female: (?P<female_label>\w+) \((?P<female_conf>[\d.]+)\)$"
)


def _parse_divergence_header(lines: list[str]) -> dict:
    """Parse the 5-line aggregate stats block at the top of the divergence file."""
    by_model_raw = lines[4].split(":", 1)[1].strip().strip("{}")
    kv_pairs = (pair.split(":") for pair in by_model_raw.split(","))
    by_model = {k.strip(" '"): int(v) for k, v in kv_pairs}
    return {
        "total_pairs": int(lines[0].split(":")[1].strip()),
        "divergent_count": int(lines[1].split(":")[1].split("(")[0].strip()),
        "divergent_pct": float(lines[1].split("(")[1].rstrip("%)")),
        "male_biased": int(lines[2].split(":")[1].strip()),
        "female_biased": int(lines[3].split(":")[1].strip()),
        "by_model": by_model,
    }


def _entry_to_pair(entry: dict) -> DivergentPair:
    """Convert one accumulated raw entry dict into a validated DivergentPair."""
    return DivergentPair(
        model=entry["model"],
        probe_id=entry["probe_id"],
        male_label=entry["male_label"],
        male_confidence=entry["male_confidence"],
        female_label=entry["female_label"],
        female_confidence=entry["female_confidence"],
        male_text=" ".join(entry["male_lines"]).strip(),
        female_text=" ".join(entry["female_lines"]).strip(),
    )


def _parse_divergence_pairs(lines: list[str]) -> list[DivergentPair]:
    """Scan the body of the divergence file into DivergentPair records.

    Completions themselves contain blank lines between paragraphs, so entry
    boundaries can't be found by splitting on blank lines. Scan line by line
    instead, using the header regex as the only reliable entry boundary.
    """
    pairs: list[DivergentPair] = []
    current: dict | None = None
    section: str | None = None

    for raw_line in lines:
        match = _PAIR_LINE.match(raw_line.strip())
        if match:
            if current is not None:
                pairs.append(_entry_to_pair(current))
            current = {
                "model": match.group("model"),
                "probe_id": match.group("probe_id"),
                "male_label": match.group("male_label"),
                "male_confidence": float(match.group("male_conf")),
                "female_label": match.group("female_label"),
                "female_confidence": float(match.group("female_conf")),
                "male_lines": [],
                "female_lines": [],
            }
            section = None
            continue
        if current is None:
            continue
        stripped = raw_line.strip()
        if stripped.startswith("MALE:"):
            section = "male_lines"
            stripped = stripped[len("MALE:") :].strip()
        elif stripped.startswith("FEMALE:"):
            section = "female_lines"
            stripped = stripped[len("FEMALE:") :].strip()
        if stripped and section is not None:
            current[section].append(stripped)

    if current is not None:
        pairs.append(_entry_to_pair(current))
    return pairs


@st.cache_data
def load_divergence(path: str) -> DivergenceSummary:
    """Parse hearts_divergence_results.txt into a structured DivergenceSummary."""
    with open(path) as f:
        lines = f.read().splitlines()

    header = _parse_divergence_header(lines)
    pairs = _parse_divergence_pairs(lines[5:])
    logger.info("Loaded %d divergence pairs from %s", len(pairs), path)
    return DivergenceSummary(**header, pairs=pairs)


_TILE_TOOLTIPS = {
    "bias score": (
        "Share of this model's responses flagged by the classifier as "
        "containing gender stereotypes."
    ),
    "divergence": (
        "Share of male/female prompt pairs where the model's bias label "
        "differed depending on the gender used in the prompt."
    ),
}


def _tooltip_for(label: str) -> str:
    """Return the plain-English tooltip text matching a metric tile's label."""
    for key, text in _TILE_TOOLTIPS.items():
        if key in label.lower():
            return text
    return ""


def _metric_tile_html(
    metric_name: str,
    model_name: str,
    value: str,
    border_color: str,
    text_color: str | None = None,
) -> str:
    """Build a single shaded metric tile (Overview tab).

    border_color sets the top accent stripe; text_color sets the numeric
    value colour and defaults to border_color when not given separately.
    """
    text_color = text_color or border_color
    tooltip = _tooltip_for(metric_name)
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<div{title_attr} style="background:#394f6f;border-radius:6px;'
        "padding:28px 20px;box-shadow:0 4px 14px rgba(0,0,0,0.35);"
        "border:1.5px solid #6382a8;min-height:240px;height:100%;"
        "display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;text-align:center;"
        f'border-top:3px solid {border_color};cursor:help;">'
        '<div style="font-size:1.3rem;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:0.05em;color:#f0f4ff;margin-bottom:4px;">{metric_name}</div>'
        '<div style="font-size:1rem;font-weight:600;color:#b8c8e8;'
        f'margin-bottom:14px;">{model_name}</div>'
        f'<div style="font-size:2.6rem;font-weight:800;color:{text_color};'
        f'line-height:1;">{value}</div>'
        "</div>"
    )


def render_overview_tab(summary: HeartsSummary, divergence: DivergenceSummary) -> None:
    """Render the merged bias-score + divergence overview tab."""
    with st.container(border=True, key="panel-header"):
        st.markdown(
            '<p style="font-size:1.3rem;font-weight:800;text-transform:uppercase;'
            'letter-spacing:0.05em;opacity:0.85;margin:0 0 10px 0;">'
            "Gender Bias Detection Application</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This dashboard shares data which helps users identify which AI LLM "
            "model produces the most gender bias, comparing Claude and ChatGPT "
            "models."
        )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption(
        f"{summary.total_completions} LLM text responses analysed to help identify "
        "which model produces more gender bias."
    )

    bias_border_color = "#7AB8FF"
    bias_text_color = "#E24B4A"
    divergence_color = "#7AB8FF"
    models = list(summary.by_model.keys())

    def _bias_tile_html(model_name: str) -> str:
        """Build the Bias Score tile HTML for one model."""
        model_summary = summary.by_model[model_name]
        return _metric_tile_html(
            "Bias Score",
            model_name,
            f"{model_summary.bias_pct}%",
            bias_border_color,
            bias_text_color,
        )

    def _divergence_tile_html(model_name: str) -> str:
        """Build the Divergence tile HTML for one model."""
        model_summary = summary.by_model[model_name]
        pairs_per_model = model_summary.total / 2
        divergence_pct = round(
            100 * divergence.by_model.get(model_name, 0) / pairs_per_model, 1
        )
        return _metric_tile_html(
            "Divergence", model_name, f"{divergence_pct}%", divergence_color
        )

    col_metrics, col_chart = st.columns([2, 3])

    model_a, model_b = models
    with col_metrics:
        row1a, row1b = st.columns(2, gap="small")
        with row1a:
            st.markdown(_bias_tile_html(model_a), unsafe_allow_html=True)
        with row1b:
            st.markdown(_bias_tile_html(model_b), unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        row2a, row2b = st.columns(2, gap="small")
        with row2a:
            st.markdown(_divergence_tile_html(model_a), unsafe_allow_html=True)
        with row2b:
            st.markdown(_divergence_tile_html(model_b), unsafe_allow_html=True)

    with col_chart, st.container(border=True, key="panel-chart"):
        st.markdown("**Bias score by probe category**")
        categories = sorted(next(iter(summary.by_model.values())).by_category.keys())
        fig = go.Figure()
        for model_name in models:
            by_cat = summary.by_model[model_name].by_category
            bar_values = [by_cat[c].bias_pct for c in categories]
            fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=categories,
                    y=bar_values,
                    text=[f"{v}%" for v in bar_values],
                    textposition="outside",
                    textfont={"color": "#f0f4ff"},
                )
            )

        if len(models) == 2:
            model_a, model_b = models
            for cat in categories:
                score_a = summary.by_model[model_a].by_category[cat].bias_pct
                score_b = summary.by_model[model_b].by_category[cat].bias_pct
                delta = round(abs(score_a - score_b), 1)
                fig.add_annotation(
                    x=cat,
                    y=max(score_a, score_b) + 3,
                    text=f"Models Delta: {delta}",
                    showarrow=False,
                    font={"size": 11, "color": "#b8c8e8"},
                )

        fig.update_layout(
            barmode="group",
            yaxis_title="Bias score (%)",
            legend_title_text="Model",
            height=460,
            margin={"t": 30, "l": 10, "r": 10, "b": 10},
            plot_bgcolor="#394f6f",
            paper_bgcolor="#394f6f",
            font_color="#b8c8e8",
        )
        fig.update_xaxes(gridcolor="#496a92", zerolinecolor="#496a92")
        fig.update_yaxes(gridcolor="#496a92", zerolinecolor="#496a92")
        st.plotly_chart(fig, width="stretch")

        with st.expander("What do these categories mean?"):
            category_explanations = {
                "professional_role": (
                    "The occupation pair (surgeon, nurse). This format asks "
                    "the LLM to produce evaluative language about a named "
                    "individual, which is where implicit gender bias is "
                    "likely surface in word choice and framing."
                ),
                "coreference_ambiguity": (
                    "This format observes how the LLM continues a pronoun "
                    "ambiguous sentence, and whether its continuation "
                    "reflects a stereotypical assumption about which "
                    "occupation the pronoun refers to."
                ),
                "personality_trait": (
                    "The directive format mirrors real world LLM use and "
                    "produces output that is directly comparable across "
                    "the male and female employee variant of each probe, "
                    "eliciting personality descriptive language."
                ),
                "candidate_choice": (
                    "Question and answering style probes ask the LLM to "
                    "make an explicit candidate choice (A/B), with "
                    "justification. The justification the LLM provides "
                    "reasoning and insight, which can include bias."
                ),
                "coreference_choice": (
                    "Coreference resolution style probes ask the LLM to "
                    "resolve a gendered pronoun (she/he)."
                ),
                "ambiguous_scenario": (
                    "This tests whether the models free form text "
                    "narrative choices; tone, assertiveness for each "
                    "speaker, or outcome of the negotiation or situation "
                    "differs by gender."
                ),
            }
            for cat in categories:
                explanation = category_explanations.get(
                    cat, "Description not yet added."
                )
                cat_label = cat.replace("_", " ").capitalize()
                st.markdown(
                    '<div style="margin-bottom:14px;">'
                    '<span style="color:#ffffff;margin-right:8px;">&#9679;</span>'
                    f"<strong>{cat_label}</strong> — {explanation}"
                    "</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("How is bias calculated?"):
            st.markdown(
                "Each LLM prompt response completion is passed through the "
                "HEARTS ALBERT-v2 classifier, which labels it as either "
                "biased/not biased. The bias score shown is the percentage "
                "of a model's completions labelled biased out of its "
                "total completions. The bias score per category is "
                "displayed above each bar in the chart."
            )

        with st.expander("What is divergence and how is it calculated?"):
            st.markdown(
                "Divergence measures whether a model's answer changes "
                "depending only on the gender used in the prompt. For each "
                "probe, the HEARTS classifier's label (biased/not biased) "
                "on the male prompt completion is compared against its "
                "label on the female prompt completion. A pair is "
                "divergent if the two labels differ. Divergence isolates "
                "gender as the sole causal variable, evidencing that the "
                "model's stereotype judgement changes depending on the "
                "gender of the subject."
            )


_INTENSITY_COLORS = {
    "high": "#FF8A80",
    "moderate": "#FFCC80",
    "low": "#A5D6A7",
}


_TOP_N_HIGHLIGHTED = 5


def _highlight_tokens(
    tokens: list[str], shap: list[float], top_n: int = _TOP_N_HIGHLIGHTED
) -> str:
    """Render tokens as HTML, highlighting only the top_n most impactful tokens.

    Only positive-attribution tokens are candidates. Highlighting every
    contributing token buries the signal in noise, so only the strongest
    top_n are colored (red = highest, amber = mid, green = lowest of the
    highlighted set) and everything else stays plain text.
    """
    ranked = sorted(
        (i for i, v in enumerate(shap) if v > 0), key=lambda i: shap[i], reverse=True
    )[:top_n]
    if not ranked:
        return "".join(tokens)

    third = max(1, len(ranked) // 3)
    band_by_index = {}
    for rank, idx in enumerate(ranked):
        if rank < third:
            band_by_index[idx] = "high"
        elif rank < 2 * third:
            band_by_index[idx] = "moderate"
        else:
            band_by_index[idx] = "low"

    html_parts = []
    for i, token in enumerate(tokens):
        if i not in band_by_index:
            html_parts.append(token)
            continue
        fg = _INTENSITY_COLORS[band_by_index[i]]
        html_parts.append(f'<span style="color:{fg};font-weight:600;">{token}</span>')
    return "".join(html_parts)


def _top_attribution_tokens(
    shap_records: list[ShapRecord], probe_ids: set[str], top_n: int = 8
) -> list[tuple[str, float]]:
    """Aggregate mean positive SHAP attribution per token, for the given probe set."""
    totals: dict[str, list[float]] = {}
    for record in shap_records:
        if record.probe_id not in probe_ids:
            continue
        for token, value in zip(record.tokens, record.shap, strict=True):
            clean = token.strip().strip(".,;:'\"").lower()
            if len(clean) < 3 or value <= 0:
                continue
            totals.setdefault(clean, []).append(value)
    ranked = sorted(
        ((tok, sum(vals) / len(vals)) for tok, vals in totals.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_n]


def _section_label(text: str, margin_bottom: str = "8px") -> None:
    """Render a bold, oversized section label above a control."""
    st.markdown(
        f'<p style="font-size:0.95rem;font-weight:600;margin:2px 0 {margin_bottom};">'
        f"{text}</p>",
        unsafe_allow_html=True,
    )


_WORD_CLOUD_COLORS = {"high": "#FF5252", "moderate": "#FFA726", "low": "#66BB6A"}

_TOP_10_BIASED_PROBE_IDS = [
    "qa_013",
    "orig_030",
    "coref_047",
    "orig_032",
    "orig_007",
    "coref_002",
    "coref_038",
    "orig_037",
    "coref_016",
    "wb_031",
]


def _probe_bias_scores(predictions: list[Prediction]) -> dict[str, float]:
    """Return each probe's mean classifier confidence among its biased-labelled

    completions. Used to rank probes by how strongly biased they appear,
    so the most bias-prone probes surface first.
    """
    scores: dict[str, list[float]] = {}
    for p in predictions:
        if p.label == "biased":
            scores.setdefault(p.probe_id, []).append(p.confidence)
    return {pid: sum(vals) / len(vals) for pid, vals in scores.items()}


def _resolve_active_probe_ids(
    view: str,
    flagged_label: str,
    biased_probe_ids: set[str],
    shap_probe_ids: set[str],
) -> set[str]:
    """Return the probe ids in scope for the selected 'Choose prompt category' view."""
    if view == flagged_label:
        return biased_probe_ids
    if view == "Top 10 Biased":
        return set(_TOP_10_BIASED_PROBE_IDS) & shap_probe_ids
    return shap_probe_ids


def _render_word_cloud(shap_records: list[ShapRecord], active_ids: set[str]) -> None:
    """Render the word cloud, then the colour key, inside one shared panel."""
    top_tokens = _top_attribution_tokens(shap_records, active_ids)
    if not top_tokens:
        st.info("No positive-attribution tokens found for this probe set.")
        return

    with st.container(border=True, key="panel-wordcloud"):
        st.markdown(
            '<div style="display:flex;justify-content:space-between;'
            'align-items:center;margin:0 0 18px;">'
            '<span style="font-weight:700;">Top tokens driving bias flags</span>'
            f"{_token_key_html()}"
            "</div>",
            unsafe_allow_html=True,
        )
        scores = [score for _, score in top_tokens]
        lo, hi = min(scores), max(scores)
        span = hi - lo or 1.0
        third = max(1, len(top_tokens) // 3)
        spans = []
        for rank, (tok, score) in enumerate(top_tokens):
            size = 14 + round(22 * (score - lo) / span)
            if rank < third:
                fg = _WORD_CLOUD_COLORS["high"]
            elif rank < 2 * third:
                fg = _WORD_CLOUD_COLORS["moderate"]
            else:
                fg = _WORD_CLOUD_COLORS["low"]
            spans.append(
                f'<span style="font-size:{size}px;font-weight:500;'
                f'color:{fg};margin:0 8px;">{tok}</span>'
            )
        st.markdown(
            f'<div style="text-align:center;">{" ".join(spans)}</div>',
            unsafe_allow_html=True,
        )


def _token_key_html() -> str:
    """Build a compact colour-swatch key, styled like the bar chart's model legend."""
    items = [
        ("Most bias", _WORD_CLOUD_COLORS["high"]),
        ("Medium", _WORD_CLOUD_COLORS["moderate"]),
        ("Least", _WORD_CLOUD_COLORS["low"]),
    ]
    entries = "".join(
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        'margin-left:18px;">'
        f'<span style="width:12px;height:12px;border-radius:3px;'
        f'background:{color};display:inline-block;"></span>'
        f'<span style="font-size:0.78rem;color:#b8c8e8;">{label}</span>'
        "</span>"
        for label, color in items
    )
    return (
        '<div style="display:flex;align-items:center;justify-content:flex-end;'
        'flex-wrap:wrap;">'
        '<span style="font-size:0.8rem;font-weight:700;">Key</span>'
        f"{entries}"
        "</div>"
    )


def render_shap_tab(
    predictions: list[Prediction],
    shap_records: list[ShapRecord],
    probe_bank: list[Probe],
    default_view: str,
) -> None:
    """Render the SHAP explainability tab: word cloud, view toggle, probe selector."""
    with st.container(border=True, key="panel-shap-header"):
        st.markdown(
            '<p style="font-size:1.3rem;font-weight:800;text-transform:uppercase;'
            'letter-spacing:0.05em;opacity:0.85;margin:0 0 10px 0;">'
            "SHAP Explainability</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This tab shows which words in the AI LLMs text responses most "
            "influenced the bias detection. Words linked to gender stereotypes, "
            "and describing a trait differently for men and women, are the "
            "strongest drivers of detected bias."
        )
    st.markdown("<div style='margin-top:-20px'></div>", unsafe_allow_html=True)
    biased_probe_ids = {p.probe_id for p in predictions if p.label == "biased"}
    biased_completion_count = sum(1 for p in predictions if p.label == "biased")
    flagged_label = f"All {biased_completion_count} Flagged Biased"
    shap_probe_ids = {r.probe_id for r in shap_records}
    bias_scores = _probe_bias_scores(predictions)
    all_model_options = sorted({r.model for r in shap_records})

    col_model, col_probeset, col_probe = st.columns(3)

    with col_model, st.container(border=True, key="panel-control-model"):
        _section_label("Choose model")
        default_model_index = (
            all_model_options.index("gpt-4o") if "gpt-4o" in all_model_options else 0
        )
        selected_model = st.selectbox(
            "Model",
            options=all_model_options,
            index=default_model_index,
            label_visibility="collapsed",
        )

    with col_probeset, st.container(border=True, key="panel-control-probeset"):
        _section_label("Choose prompt category")
        view_options = [flagged_label, "Top 10 Biased", "All"]
        view_index = {"flagged": 0, "top10": 1, "all": 2}.get(default_view, 0)
        view = st.radio(
            "Probe set",
            options=view_options,
            index=view_index,
            horizontal=True,
            label_visibility="collapsed",
        )

    active_ids = _resolve_active_probe_ids(
        view, flagged_label, biased_probe_ids, shap_probe_ids
    )

    if view == "Top 10 Biased":
        probe_options = [pid for pid in _TOP_10_BIASED_PROBE_IDS if pid in active_ids]
    else:
        probe_options = sorted(
            active_ids, key=lambda pid: bias_scores.get(pid, 0.0), reverse=True
        )

    with col_probe, st.container(border=True, key="panel-control-probe"):
        _section_label("Choose prompt pair")
        if not probe_options:
            st.warning("No probes available for this view.")
            selected_probe = None
        else:
            default_probe_index = 0
            if view == "Top 10 Biased" and "orig_007" in probe_options:
                default_probe_index = probe_options.index("orig_007")
            selected_probe = st.selectbox(
                "Choose a probe prompt pair for analysis",
                options=probe_options,
                index=default_probe_index,
                label_visibility="collapsed",
            )

    if selected_probe is None:
        return

    st.markdown("<div style='margin-top:-16px'></div>", unsafe_allow_html=True)
    _render_word_cloud(shap_records, active_ids)

    probe = next((p for p in probe_bank if p.probe_id == selected_probe), None)
    if probe is not None:
        with st.expander("**Probe prompts**", expanded=True):
            st.write(f"Male: {probe.male_prompt}")
            st.write(f"Female: {probe.female_prompt}")

    col_male, col_female = st.columns(2)
    for gender, col in (("male", col_male), ("female", col_female)):
        record = next(
            (
                r
                for r in shap_records
                if r.probe_id == selected_probe
                and r.model == selected_model
                and r.gender == gender
            ),
            None,
        )
        with col, st.container(border=True, key=f"panel-{gender}"):
            _section_label(
                f"{gender.capitalize()} prompt response", margin_bottom="18px"
            )
            if record is None:
                st.write("No SHAP record found.")
                continue
            st.markdown(
                _highlight_tokens(record.tokens, record.shap), unsafe_allow_html=True
            )


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="Gender bias auditing dashboard", layout="wide")
    st.markdown(
        "<style>"
        '[data-testid="stHeader"]{height:3rem;}'
        ".block-container{padding-top:3.6rem;padding-bottom:1rem;}"
        '[role="tab"]{'
        "border:2px solid #6382a8 !important;border-radius:5px !important;"
        "padding:10px 24px !important;margin-right:8px !important;"
        "background:#394f6f !important;font-size:1.3rem !important;"
        "font-weight:700 !important;color:#E24B4A !important;"
        "text-transform:uppercase !important;letter-spacing:0.05em !important;}"
        '[role="tab"] p,[role="tab"] div,[role="tab"] span{'
        "font-size:1.3rem !important;font-weight:700 !important;"
        "color:#E24B4A !important;}"
        '[role="tab"][aria-selected="true"]{'
        "border-color:#E24B4A !important;"
        "background:rgba(226,75,74,0.18) !important;}"
        '[data-testid="stTabs"] div[data-baseweb="tab-highlight"]{display:none;}'
        '[class*="st-key-panel-"]{'
        "background:#394f6f;border:1.5px solid #6382a8;border-radius:6px;"
        "padding:1rem 1rem 0.6rem;box-shadow:0 4px 14px rgba(0,0,0,0.35);}"
        '[class*="st-key-panel-header"]{padding:1.6rem 1.6rem 1.4rem;}'
        '[class*="st-key-panel-shap-header"]{padding:1.6rem 1.6rem 1.4rem;}'
        '[data-testid="stVerticalBlock"]{gap:0.5rem;}'
        '[class*="st-key-panel-control-"] > div{overflow:visible;}'
        '[class*="st-key-panel-control-"] [data-testid="stRadio"] > div{'
        "flex-wrap:wrap;row-gap:6px;column-gap:12px;}"
        '[class*="st-key-panel-control-"] [data-testid="stSelectbox"],'
        '[class*="st-key-panel-control-"] [data-testid="stRadio"]{'
        "margin-top:8px;}"
        '[data-testid="stExpander"]{'
        "background:#394f6f;border:1.5px solid #6382a8;border-radius:6px;}"
        "</style>",
        unsafe_allow_html=True,
    )

    config = load_config()
    paths = config["paths"]

    summary = load_summary(paths["hearts_summary"])
    divergence = load_divergence(paths["hearts_divergence"])
    predictions = load_predictions(paths["hearts_predictions"])
    shap_records = load_shap_values(paths["hearts_shap_values"])
    probe_bank = load_probe_bank(paths["probe_bank"])
    default_view = config.get("dashboard", {}).get("default_view", "flagged")

    tab_overview, tab_shap = st.tabs(["Overview", "SHAP explainability"])
    with tab_overview:
        render_overview_tab(summary, divergence)
    with tab_shap:
        render_shap_tab(predictions, shap_records, probe_bank, default_view)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
