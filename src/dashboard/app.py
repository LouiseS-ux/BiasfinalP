"""
Stage 5 — Streamlit Dashboard.

Read-only browser interface. Loads from data/*.json + hearts_divergence_results.txt
only. No database, no writes. All charts use Plotly. Uses st.cache_data on all
file loads.

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

# ---------------------------------------------------------------------------
# Data models — match the real pipeline output schemas exactly.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data loading — all reads are cached and read-only, per coding_standards.md.
# ---------------------------------------------------------------------------


@st.cache_data
def load_config() -> dict:
    """Load pipeline configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_predictions(path: str) -> list[Prediction]:
    """Load and validate hearts_predictions.json."""
    with open(path) as f:
        raw = json.load(f)
    return [Prediction(**row) for row in raw]


@st.cache_data
def load_shap_values(path: str) -> list[ShapRecord]:
    """Load and validate hearts_shap_values.json."""
    with open(path) as f:
        raw = json.load(f)
    return [ShapRecord(**row) for row in raw]


@st.cache_data
def load_probe_bank(path: str) -> list[Probe]:
    """Load and validate probe_bank.json."""
    with open(path) as f:
        raw = json.load(f)
    return [Probe(**row) for row in raw]


@st.cache_data
def load_summary(path: str) -> HeartsSummary:
    """Load and validate hearts_summary.json."""
    with open(path) as f:
        raw = json.load(f)
    return HeartsSummary(**raw)


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
    return DivergenceSummary(**header, pairs=pairs)


# ---------------------------------------------------------------------------
# Tab 1 — overview (bias score + divergence, merged)
# ---------------------------------------------------------------------------

# Plain-English explanations shown as tooltips on the metric tiles.
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


def _metric_tile_html(label: str, value: str, accent: str) -> str:
    """Build a single shaded, accent-colored metric tile (Overview tab)."""
    tooltip = _tooltip_for(label)
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<div{title_attr} style="background:rgba(130,130,150,0.14);border-radius:4px;'
        "padding:14px 16px;box-shadow:0 2px 6px rgba(0,0,0,0.12);"
        f'border-top:4px solid {accent};height:100%;cursor:help;">'
        '<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.04em;opacity:0.75;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:1.7rem;font-weight:700;color:{accent};'
        f'line-height:1.1;">{value}</div>'
        "</div>"
    )


def render_overview_tab(summary: HeartsSummary, divergence: DivergenceSummary) -> None:
    """Render the merged bias-score + divergence overview tab."""
    with st.container(border=True, key="panel-header"):
        st.markdown(
            '<p style="font-size:1.15rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.05em;opacity:0.85;margin:0 0 4px 0;">'
            "Gender Bias Detection Application</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This dashboard shares data which helps users identify which AI LLM "
            "model produces the most gender bias, comparing Claude and ChatGPT "
            "models."
        )
    st.caption(
        f"{summary.total_completions} LLM text responses analysed to help identify "
        "which model produces more gender bias."
    )

    bias_color = "#5DADE2"
    divergence_color = "#FF5722"
    models = list(summary.by_model.keys())
    tiles = []
    for model_name in models:
        model_summary = summary.by_model[model_name]
        divergence_pct = round(
            100 * divergence.by_model.get(model_name, 0) / model_summary.total, 1
        )
        tiles.append(
            (f"{model_name} — bias score", f"{model_summary.bias_pct}%", bias_color)
        )
        tiles.append(
            (f"{model_name} — divergence", f"{divergence_pct}%", divergence_color)
        )

    cols = st.columns(len(tiles))
    for col, (label, value, accent) in zip(cols, tiles, strict=True):
        with col:
            st.markdown(_metric_tile_html(label, value, accent), unsafe_allow_html=True)

    # Spacer — gives the tiles room to breathe above the chart panel.
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    with st.container(border=True, key="panel-chart"):
        st.markdown("**Bias score by probe category**")
        categories = sorted(next(iter(summary.by_model.values())).by_category.keys())
        fig = go.Figure()
        for model_name in models:
            by_cat = summary.by_model[model_name].by_category
            fig.add_trace(
                go.Bar(
                    name=model_name,
                    x=categories,
                    y=[by_cat[c].bias_pct for c in categories],
                )
            )

        # Delta annotation — shows the gap between the two models' bias
        # scores per category, directly above the taller of the two bars.
        if len(models) == 2:
            model_a, model_b = models
            for cat in categories:
                score_a = summary.by_model[model_a].by_category[cat].bias_pct
                score_b = summary.by_model[model_b].by_category[cat].bias_pct
                delta = round(abs(score_a - score_b), 1)
                fig.add_annotation(
                    x=cat,
                    y=max(score_a, score_b) + 3,
                    text=f"Δ {delta}",
                    showarrow=False,
                    font={"size": 11, "color": "gray"},
                )

        fig.update_layout(
            barmode="group",
            yaxis_title="Bias score (%)",
            legend_title_text="Model",
            height=280,
            margin={"t": 30, "l": 10, "r": 10, "b": 10},
        )
        st.plotly_chart(fig, width="stretch")

        # Plain-English explanation of each probe category, shown collapsed
        # underneath the chart so the bar labels stay uncluttered.
        with st.expander("What do these categories mean?"):
            category_explanations = {
                # TODO: fill in with your actual category keys from
                # probe_bank.json and a one-line description of each.
                # e.g. "hiring": "Prompts about recommending a job candidate.",
            }
            for cat in categories:
                explanation = category_explanations.get(
                    cat, "Description not yet added."
                )
                st.markdown(f"**{cat}** — {explanation}")

        with st.expander("How is bias calculated?"):
            st.markdown(
                "Each LLM completion is passed through the HEARTS ALBERT-v2 "
                "classifier, which labels it as either **biased** or "
                "**not biased**. The bias score shown above is the "
                "percentage of a model's completions labelled biased out "
                "of its total completions, per category or overall."
            )

        with st.expander("What is divergence and how is it calculated?"):
            st.markdown(
                "Divergence measures whether a model's answer changes "
                "depending only on the gender used in the prompt. For "
                "matched male/female prompt pairs where the model made a "
                "clear choice both times, a pair is **divergent** if the "
                "male-prompt and female-prompt answers differ. The "
                "divergence score is the percentage of such pairs that are "
                "divergent, out of all pairs where a choice was made in "
                "both versions."
            )


# ---------------------------------------------------------------------------
# Tab 2 — SHAP explainability
# ---------------------------------------------------------------------------

_INTENSITY_COLORS = {
    "high": ("#FF8A80", "#B71C1C"),
    "moderate": ("#FFCC80", "#E65100"),
    "low": ("#A5D6A7", "#1B5E20"),
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
        bg, fg = _INTENSITY_COLORS[band_by_index[i]]
        html_parts.append(
            f'<span style="background:{bg};color:{fg};padding:1px 3px;'
            f'border-radius:3px;">{token}</span>'
        )
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


def _section_label(text: str) -> None:
    """Render a bold, oversized section label above a control."""
    st.markdown(
        f'<p style="font-size:0.95rem;font-weight:600;margin:2px 0 0;">{text}</p>',
        unsafe_allow_html=True,
    )


_WORD_CLOUD_COLORS = {"high": "#FF5252", "moderate": "#FFA726", "low": "#66BB6A"}


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
    biased_probe_ids: set[str],
    shap_probe_ids: set[str],
    bias_scores: dict[str, float],
) -> set[str]:
    """Return the probe ids in scope for the selected 'Choose prompt category' view."""
    if view == "All Flagged Biased":
        return biased_probe_ids
    if view == "Top 5 biased":
        ranked = sorted(
            biased_probe_ids & shap_probe_ids,
            key=lambda pid: bias_scores.get(pid, 0.0),
            reverse=True,
        )
        return set(ranked[:5])
    return shap_probe_ids


def _render_word_cloud(shap_records: list[ShapRecord], active_ids: set[str]) -> None:
    """Render the top-token word cloud panel for the given probe set, or a fallback."""
    top_tokens = _top_attribution_tokens(shap_records, active_ids)
    if not top_tokens:
        st.info("No positive-attribution tokens found for this probe set.")
        return

    with st.container(border=True, key="panel-wordcloud"):
        st.markdown("**Top tokens driving bias flags**")
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


def render_shap_tab(
    predictions: list[Prediction],
    shap_records: list[ShapRecord],
    probe_bank: list[Probe],
    default_view: str,
) -> None:
    """Render the SHAP explainability tab: word cloud, view toggle, probe selector."""
    with st.container(border=True, key="panel-shap-header"):
        st.markdown(
            '<p style="font-size:1.15rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.05em;opacity:0.85;margin:0 0 4px 0;">'
            "SHAP Explainability</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This tab shows which words in the AI LLMs text responses most "
            "influenced the bias detection. Words linked to gender stereotypes, "
            "like describing a trait differently for men and women, are the "
            "strongest drivers of detected bias."
        )
    legend_items = [
        ("Green = least", _WORD_CLOUD_COLORS["low"]),
        ("Orange = medium", _WORD_CLOUD_COLORS["moderate"]),
        ("Red = most bias signal", _WORD_CLOUD_COLORS["high"]),
    ]
    legend_html = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="color:{color};font-weight:600;">&#9679; {label}</span>'
        for label, color in legend_items
    )

    biased_probe_ids = {p.probe_id for p in predictions if p.label == "biased"}
    # "All" means all probes with SHAP data available, not all probes in the
    # probe bank — SHAP may only be computed for a subset (see coding_standards.md,
    # stage4_shap.py: "limit initial runs to high-confidence predictions").
    shap_probe_ids = {r.probe_id for r in shap_records}
    bias_scores = _probe_bias_scores(predictions)
    all_model_options = sorted({r.model for r in shap_records})

    col_model, col_probeset, col_probe = st.columns(3)

    with col_model, st.container(border=True, key="panel-control-model"):
        _section_label("Choose model")
        selected_model = st.selectbox(
            "Model", options=all_model_options, label_visibility="collapsed"
        )

    with col_probeset, st.container(border=True, key="panel-control-probeset"):
        _section_label("Choose prompt category")
        view_options = ["All Flagged Biased", "Top 5 biased", "All"]
        view_index = {"flagged": 0, "all": 2}.get(default_view, 0)
        view = st.radio(
            "Probe set",
            options=view_options,
            index=view_index,
            horizontal=True,
            label_visibility="collapsed",
        )

    active_ids = _resolve_active_probe_ids(
        view, biased_probe_ids, shap_probe_ids, bias_scores
    )

    # Most strongly biased probes first, so the viewer sees the clearest
    # examples of bias without having to hunt through the list.
    probe_options = sorted(
        active_ids, key=lambda pid: bias_scores.get(pid, 0.0), reverse=True
    )

    with col_probe, st.container(border=True, key="panel-control-probe"):
        _section_label("Choose prompt pair")
        if not probe_options:
            st.warning("No probes available for this view.")
            selected_probe = None
        else:
            selected_probe = st.selectbox(
                "Choose a probe prompt pair for analysis",
                options=probe_options,
                label_visibility="collapsed",
            )

    if selected_probe is None:
        return

    with st.container(border=True, key="panel-legend"):
        st.caption(
            "Tokens inside words driving bias - coloured in order of SHAP signal "
            "amount."
        )
        st.markdown(legend_html, unsafe_allow_html=True)

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
            _section_label(f"{gender.capitalize()} prompt response")
            if record is None:
                st.write("No SHAP record found.")
                continue
            st.markdown(
                _highlight_tokens(record.tokens, record.shap), unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="Gender bias auditing dashboard", layout="wide")
    st.markdown(
        "<style>"
        # Extra top padding stops the browser/Streamlit toolbar from
        # overlapping the tab navigation buttons.
        '[data-testid="stHeader"]{height:3rem;}'
        ".block-container{padding-top:3.6rem;padding-bottom:1rem;}"
        '[data-testid="stTabs"] button p{font-size:0.85rem;font-weight:600;'
        "text-transform:uppercase;letter-spacing:0.05em;opacity:0.85;}"
        '[data-testid="stTabs"] button{'
        # Square-ish corners (was 10px) to match the reference design.
        "border:1.5px solid rgba(130,130,150,0.4);border-radius:4px;"
        "padding:6px 18px;margin-right:8px;background:rgba(130,130,150,0.12);}"
        '[data-testid="stTabs"] button[aria-selected="true"]{'
        "border-color:#FF5722;background:rgba(255,87,34,0.16);}"
        '[data-testid="stTabs"] div[data-baseweb="tab-highlight"]{display:none;}'
        '[class*="st-key-panel-"]{'
        "background:rgba(130,130,150,0.10);border-radius:8px;padding:1rem 1rem 0.6rem;"
        "box-shadow:0 2px 6px rgba(0,0,0,0.10);}"
        '[data-testid="stVerticalBlock"]{gap:0.5rem;}'
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
