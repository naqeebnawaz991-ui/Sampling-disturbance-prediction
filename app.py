from __future__ import annotations

from pathlib import Path
import io
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Soil Sampling Disturbance Predictor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "AreaRatio",
    "CuttingEdge",
    "PlasticityIndex",
    "OCR",
    "VerticalStress",
]

FEATURE_LABELS = {
    "AreaRatio": "Area ratio, AR",
    "CuttingEdge": "Cutting-edge angle, CE",
    "PlasticityIndex": "Plasticity index, PI",
    "OCR": "Overconsolidation ratio, OCR",
    "VerticalStress": "In-situ vertical effective stress, σ′ᵥ",
}

FEATURE_UNITS = {
    "AreaRatio": "%",
    "CuttingEdge": "°",
    "PlasticityIndex": "%",
    "OCR": "–",
    "VerticalStress": "kPa",
}

SHORT_LABELS = {
    "AreaRatio": "AR [%]",
    "CuttingEdge": "CE [°]",
    "PlasticityIndex": "PI [%]",
    "OCR": "OCR",
    "VerticalStress": "σ′ᵥ [kPa]",
}

FEATURE_HELP = {
    "AreaRatio": "Sampler wall area ratio (ASTM D1587-style). Lower = thinner-walled, "
                 "generally less soil-displacing sampler.",
    "CuttingEdge": "Cutting-edge taper angle of the sampler tip. Sharper (smaller) "
                   "angles generally cause less disturbance on penetration.",
    "PlasticityIndex": "Index plasticity of the soil (Liquid Limit − Plastic Limit).",
    "OCR": "Ratio of maximum past effective vertical stress to current in-situ "
           "effective vertical stress.",
    "VerticalStress": "Effective overburden (vertical) stress at the sample depth.",
}

REQUIRED_FILES = [
    "best_model.joblib",
    "applicability_domain.xlsx",
]

OPTIONAL_FILES = [
    "best_model_summary.xlsx",
    "reproducibility_settings.xlsx",
    "software_versions.json",
    "best_model_parameters.json",
]


# =============================================================================
# DESIGN TOKENS
# =============================================================================
INK = "#1B2430"
INK_SOFT = "#4C5567"
PAPER = "#F7F8FA"
PANEL = "#FFFFFF"
LINE = "#E3E6EC"

PRIMARY = "#22405B"      # deep slate navy — headers, primary actions
ACCENT = "#3E7C8C"       # muted teal — interactive accents
GOOD = "#2F7A4F"         # in-domain / excellent
WARN = "#B0791A"         # caution / poor
BAD = "#B23A2E"          # extrapolation / very poor
NEUTRAL = "#8A8F9C"       # not-classified / muted

FONT_DISPLAY = "'Source Serif 4', Georgia, serif"
FONT_BODY = "'Inter', -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'JetBrains Mono', monospace"

QUALITY_COLORS = {
    "Very good to excellent": GOOD,
    "Good to fair": "#5C8F3A",
    "Poor": WARN,
    "Very poor": BAD,
    "Not classified": NEUTRAL,
}

STATUS_COLORS = {
    "status-good": GOOD,
    "status-warn": WARN,
    "status-bad": BAD,
}


# =============================================================================
# STYLING
# =============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: {FONT_BODY}; color: {INK}; }}
    .stApp {{ background: {PAPER}; }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Masthead ---------------------------------------------------------*/
    .sd-masthead {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 2px solid {PRIMARY};
        padding-bottom: 0.7rem;
        margin-bottom: 0.35rem;
    }}
    .sd-masthead-title {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        font-size: 2.05rem;
        color: {PRIMARY};
        margin: 0;
        letter-spacing: -0.01em;
    }}
    .sd-masthead-sub {{
        font-family: {FONT_MONO};
        font-size: 0.76rem;
        color: {INK_SOFT};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        white-space: nowrap;
    }}
    .sd-tagline {{
        color: {INK_SOFT};
        font-size: 0.95rem;
        margin: 0.35rem 0 1.1rem 0;
    }}
    .sd-eyebrow {{
        font-family: {FONT_MONO};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {ACCENT};
        margin: 0.2rem 0 0.5rem 0;
        font-weight: 600;
    }}

    /* Cards --------------------------------------------------------------*/
    .sd-card {{
        background: {PANEL};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
    }}

    /* Status banners -----------------------------------------------------*/
    .status-good, .status-warn, .status-bad {{
        padding: 1rem 1.15rem;
        border-radius: 8px;
        margin: 0.6rem 0 0.5rem 0;
        font-size: 0.94rem;
    }}
    .status-good {{ border-left: 5px solid {GOOD}; background: {GOOD}12; }}
    .status-warn {{ border-left: 5px solid {WARN}; background: {WARN}14; }}
    .status-bad  {{ border-left: 5px solid {BAD};  background: {BAD}12; }}
    .status-label {{
        font-family: {FONT_MONO};
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.78rem;
        display: block;
        margin-bottom: 0.25rem;
    }}

    /* Pills / badges -------------------------------------------------------*/
    .sd-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: {FONT_MONO};
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.32rem 0.75rem;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .sd-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

    /* Metric cards ---------------------------------------------------------*/
    div[data-testid="stMetric"] {{
        border: 1px solid {LINE};
        padding: 0.95rem 1rem;
        border-radius: 10px;
        background: {PANEL};
        min-height: 108px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {INK_SOFT};
        font-family: {FONT_MONO};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        color: {PRIMARY};
        font-family: {FONT_DISPLAY};
        font-weight: 700;
    }}

    /* Inputs -----------------------------------------------------------*/
    div[data-testid="stNumberInput"] input {{
        background: {PAPER};
        border-radius: 6px;
        font-family: {FONT_MONO};
        font-size: 1.08rem;
        font-weight: 600;
        color: {PRIMARY};
        padding: 0.5rem 0.6rem;
    }}
    .sd-input-caption {{
        font-family: {FONT_MONO};
        font-size: 0.7rem;
        color: {INK_SOFT};
        margin-top: 0.4rem;
        line-height: 1.4;
    }}
    .sd-panel-title {{
        font-family: {FONT_MONO};
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {ACCENT};
        border-bottom: 2px solid {ACCENT}55;
        padding-bottom: 0.4rem;
        margin-bottom: 0.7rem;
    }}
    .sd-field-label {{
        font-weight: 600;
        font-size: 0.86rem;
        color: {INK};
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }}
    .sd-field-unit {{
        font-family: {FONT_MONO};
        font-size: 0.72rem;
        color: {INK_SOFT};
        font-weight: 500;
        text-transform: none;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 10px !important;
    }}

    /* Buttons -------------------------------------------------------------*/
    div.stButton > button[kind="primary"] {{
        background: {PRIMARY};
        min-height: 3.1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        letter-spacing: 0.02em;
    }}
    div.stButton > button[kind="primary"]:hover {{ background: {ACCENT}; }}
    .stDownloadButton > button {{
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid {PRIMARY};
        color: {PRIMARY};
        background: transparent;
    }}
    .stDownloadButton > button:hover {{
        background: {PRIMARY}; color: white;
    }}

    /* Quality box -----------------------------------------------------*/
    .quality-box {{
        border: 1px solid {LINE};
        background: {PAPER};
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        margin-top: 0.5rem;
    }}
    .quality-title {{
        color: {PRIMARY};
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: {FONT_DISPLAY};
        font-size: 1.02rem;
    }}
    .quality-row {{ font-size: 0.88rem; margin: 0.2rem 0; }}

    /* Sidebar ------------------------------------------------------------*/
    [data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {LINE};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {PRIMARY}; font-family: {FONT_DISPLAY};
    }}
    .sd-file-chip {{
        font-family: {FONT_MONO};
        font-size: 0.78rem;
        background: {PAPER};
        border: 1px solid {LINE};
        border-radius: 5px;
        padding: 0.28rem 0.55rem;
        display: block;
        margin-bottom: 0.35rem;
        color: {INK_SOFT};
    }}

    /* Tables --------------------------------------------------------------*/
    div[data-testid="stDataFrame"] {{
        border: 1px solid {LINE};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Tabs ------------------------------------------------------------------*/
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: {FONT_MONO}; font-size: 0.82rem;
        text-transform: uppercase; letter-spacing: 0.06em; color: {INK_SOFT};
    }}
    .stTabs [aria-selected="true"] {{
        color: {PRIMARY} !important; border-bottom: 2px solid {ACCENT} !important;
    }}

    h1, h2, h3 {{ font-family: {FONT_DISPLAY} !important; color: {PRIMARY} !important; }}
    hr {{ border-color: {LINE}; }}
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def pill_html(label: str, color: str) -> str:
    return (
        f'<span class="sd-pill" style="background:{color}1A; color:{color}; '
        f'border:1px solid {color}55;"><span class="sd-dot" '
        f'style="background:{color};"></span>{label}</span>'
    )


# =============================================================================
# PIPELINE OUTPUT LOADING  (unchanged from the research-pipeline contract)
# =============================================================================
def discover_pipeline_output_dir() -> Path | None:
    """Find a directory containing the exact outputs from the research pipeline."""
    candidates: list[Path] = []

    explicit = Path("pipeline_outputs")
    if explicit.is_dir():
        candidates.append(explicit)

    candidates.append(Path("."))

    results_dirs = [p for p in Path(".").glob("Results_*") if p.is_dir()]
    results_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates.extend(results_dirs)

    for folder in candidates:
        if all((folder / name).exists() for name in REQUIRED_FILES):
            return folder.resolve()
    return None


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_pipeline_outputs(folder_str: str):
    folder = Path(folder_str)

    ranges = pd.read_excel(
        folder / "applicability_domain.xlsx",
        sheet_name="Predictor_Ranges",
    )
    method = pd.read_excel(
        folder / "applicability_domain.xlsx",
        sheet_name="Method",
    )

    required_range_cols = {
        "Feature",
        "Minimum",
        "Percentile_1",
        "Percentile_99",
        "Maximum",
        "Mean",
        "Standard_Deviation",
    }
    missing = required_range_cols.difference(ranges.columns)
    if missing:
        raise ValueError(
            "applicability_domain.xlsx is missing expected Predictor_Ranges columns: "
            + ", ".join(sorted(missing))
        )

    missing_features = [
        f for f in FEATURES if f not in set(ranges["Feature"].astype(str))
    ]
    if missing_features:
        raise ValueError(
            "applicability_domain.xlsx does not contain all required predictors: "
            + ", ".join(missing_features)
        )

    if "Multivariate_Distance_Threshold" not in method.columns:
        raise ValueError(
            "The Method sheet in applicability_domain.xlsx does not contain "
            "Multivariate_Distance_Threshold."
        )

    optional = {}
    for name in OPTIONAL_FILES:
        path = folder / name
        if not path.exists():
            optional[name] = None
            continue
        if path.suffix.lower() == ".xlsx":
            optional[name] = pd.read_excel(path)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                optional[name] = json.load(fh)

    return ranges, method, optional


def build_domain_lookup(ranges_df: pd.DataFrame):
    lookup = {}
    indexed = ranges_df.set_index("Feature")
    for feature in FEATURES:
        row = indexed.loc[feature]
        lookup[feature] = {
            "minimum": float(row["Minimum"]),
            "p01": float(row["Percentile_1"]),
            "p99": float(row["Percentile_99"]),
            "maximum": float(row["Maximum"]),
            "mean": float(row["Mean"]),
            "std": float(row["Standard_Deviation"]),
        }
    return lookup


# =============================================================================
# APPLICABILITY DOMAIN  (unchanged math from the research pipeline)
# =============================================================================
def evaluate_applicability(input_df, domain_lookup, multivariate_threshold):
    rows = []
    strict_inside = True
    robust_inside = True
    z_values = []

    for feature in FEATURES:
        value = float(input_df.iloc[0][feature])
        d = domain_lookup[feature]

        inside_minmax = d["minimum"] <= value <= d["maximum"]
        inside_robust = d["p01"] <= value <= d["p99"]

        strict_inside = strict_inside and inside_minmax
        robust_inside = robust_inside and inside_robust

        if d["std"] <= 0 or not np.isfinite(d["std"]):
            raise ValueError(
                f"Invalid standard deviation for {feature} in applicability_domain.xlsx"
            )

        z = (value - d["mean"]) / d["std"]
        z_values.append(z)

        if not inside_minmax:
            status = "Outside observed min–max"
        elif not inside_robust:
            status = "Within min–max; outside 1st–99th percentile"
        else:
            status = "Within 1st–99th percentile"

        pct_min = 0.0 if value <= d["minimum"] else None
        rows.append(
            {
                "Parameter": SHORT_LABELS[feature],
                "Input": value,
                "Observed min": d["minimum"],
                "1st percentile": d["p01"],
                "99th percentile": d["p99"],
                "Observed max": d["maximum"],
                "Status": status,
            }
        )

    multivariate_distance = float(np.sqrt(np.square(z_values).sum()))
    inside_multivariate = multivariate_distance <= float(multivariate_threshold)

    # Same combined-domain rule as the research pipeline.
    inside_combined = robust_inside and inside_multivariate

    if not strict_inside:
        display_level = "EXTRAPOLATION"
        css = "status-bad"
        message = (
            "At least one input is outside the observed development-data range. "
            "The model can return a numerical prediction, but it represents extrapolation."
        )
    elif not inside_combined:
        display_level = "OUTSIDE COMBINED AD"
        css = "status-warn"
        reasons = []
        if not robust_inside:
            reasons.append(
                "one or more predictors are outside the 1st–99th percentile range"
            )
        if not inside_multivariate:
            reasons.append(
                "the standardized multivariate distance exceeds the pipeline threshold"
            )
        message = (
            "Prediction should be interpreted cautiously because "
            + " and ".join(reasons)
            + "."
        )
    else:
        display_level = "WITHIN COMBINED AD"
        css = "status-good"
        message = (
            "All five predictors are within their 1st–99th percentile ranges and the "
            "standardized multivariate distance is within the pipeline threshold."
        )

    return {
        "strict_inside": strict_inside,
        "robust_inside": robust_inside,
        "multivariate_distance": multivariate_distance,
        "multivariate_threshold": float(multivariate_threshold),
        "inside_multivariate": inside_multivariate,
        "inside_combined": inside_combined,
        "display_level": display_level,
        "css": css,
        "message": message,
        "feature_table": pd.DataFrame(rows),
    }


# =============================================================================
# LUNNE CLASSIFICATION AND SHAP HELPERS  (unchanged)
# =============================================================================
# Lunne, T., Berre, T. and Strandvik, S. (1997) sample-quality criteria,
# Table 1: normalized void-ratio-change (Δe/e0) thresholds, given separately
# for the OCR 1-2 and OCR 2-4 bands. Criteria are not defined by this table
# for OCR < 1 or OCR > 4, so those cases fall back to "Not classified".
LUNNE_BANDS = {
    "OCR 1-2": {
        "ocr_range": (1.0, 2.0),
        "thresholds": [
            (0.04, "Very good to excellent"),
            (0.07, "Good to fair"),
            (0.14, "Poor"),
        ],
        "very_poor_label": "Very poor",
    },
    "OCR 2-4": {
        "ocr_range": (2.0, 4.0),
        "thresholds": [
            (0.03, "Very good to excellent"),
            (0.05, "Good to fair"),
            (0.10, "Poor"),
        ],
        "very_poor_label": "Very poor",
    },
}


def _classify_against_band(prediction: float, band: dict) -> tuple[str, str]:
    """Return (label, criterion_text) for a prediction against one Lunne band."""
    lower = 0.0
    for upper, label in band["thresholds"]:
        if prediction < upper:
            if lower == 0.0:
                return label, f"Δe/e₀ < {upper:g}"
            return label, f"{lower:g} ≤ Δe/e₀ < {upper:g}"
        lower = upper
    return band["very_poor_label"], f"Δe/e₀ ≥ {lower:g}"


def classify_lunne_quality(prediction: float, ocr: float):
    """Lunne et al. (1997) sample-quality classification (Table 1), covering
    both the OCR 1-2 and OCR 2-4 bands. Outside 1 <= OCR <= 4, the source
    table does not define criteria, so the result is 'Not classified'."""
    if not np.isfinite(ocr):
        return {
            "label": "Not classified",
            "criterion": "OCR value is missing or invalid.",
            "band": None,
            "applicable": False,
        }

    for band_name, band in LUNNE_BANDS.items():
        lo, hi = band["ocr_range"]
        if lo <= ocr <= hi:
            label, criterion = _classify_against_band(prediction, band)
            return {
                "label": label,
                "criterion": f"{criterion}  (Lunne et al. 1997, {band_name})",
                "band": band_name,
                "applicable": True,
            }

    return {
        "label": "Not classified",
        "criterion": (
            f"OCR = {ocr:.2f} falls outside the 1–4 range covered by the "
            f"Lunne et al. (1997) Table 1 criteria."
        ),
        "band": None,
        "applicable": False,
    }


def format_shap_value(value: float) -> str:
    """Use 4 decimals normally and scientific notation for tiny non-zero values."""
    value = float(value)
    if not np.isfinite(value):
        return "—"
    if value == 0.0:
        return "0.0000"
    if abs(value) < 1e-4:
        return f"{value:+.2e}"
    return f"{value:+.4f}"


def make_shap_table(input_df, explanation):
    values = np.asarray(explanation.values[0], dtype=float)
    rows = []
    for i, feature in enumerate(FEATURES):
        shap_value = float(values[i])
        rows.append(
            {
                "Parameter": SHORT_LABELS[feature],
                "Input": float(input_df.iloc[0][feature]),
                "SHAP contribution": shap_value,
                "Direction": (
                    "Increases prediction"
                    if shap_value > 0
                    else "Decreases prediction"
                    if shap_value < 0
                    else "Neutral"
                ),
                "|SHAP|": abs(shap_value),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("|SHAP|", ascending=False)
        .reset_index(drop=True)
    )


def make_waterfall_figure(shap_table: pd.DataFrame, base_value: float):
    """Interactive Plotly waterfall, styled to match the app theme, replacing
    the default matplotlib shap.plots.waterfall for a more polished look."""
    ordered = shap_table.sort_values("|SHAP|", ascending=True)
    labels = list(ordered["Parameter"]) 
    values = list(ordered["SHAP contribution"])

    measures = ["relative"] * len(values)
    x_labels = ["Baseline"] + labels + ["Prediction"]
    y_values = [base_value] + values + [0]
    measures_full = ["absolute"] + measures + ["total"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures_full,
        x=x_labels,
        y=y_values,
        connector=dict(line=dict(color=LINE, width=1)),
        increasing=dict(marker=dict(color=BAD)),
        decreasing=dict(marker=dict(color=ACCENT)),
        totals=dict(marker=dict(color=PRIMARY)),
        text=[f"{v:+.3f}" if i not in (0, len(y_values) - 1) else f"{v:.3f}"
              for i, v in enumerate(y_values)],
        textposition="outside",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color=INK, size=12),
        yaxis=dict(title="Predicted SampleDisturbance (Δe/e₀)", gridcolor=LINE, zerolinecolor=INK_SOFT),
        xaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def make_shap_bar_figure(shap_table: pd.DataFrame):
    ordered = shap_table.sort_values("|SHAP|", ascending=True)
    colors = [BAD if v > 0 else ACCENT for v in ordered["SHAP contribution"]]
    fig = go.Figure(go.Bar(
        x=ordered["SHAP contribution"],
        y=ordered["Parameter"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in ordered["SHAP contribution"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=50, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color=INK, size=12),
        xaxis=dict(title="Impact on predicted disturbance", zeroline=True,
                   zerolinecolor=INK_SOFT, gridcolor=LINE),
        yaxis=dict(title=""),
    )
    return fig


def lunne_table_html(active_band: str | None, active_label: str | None) -> str:
    """Render Lunne et al. (1997) Table 1 in full (both OCR bands), highlighting
    the column for the currently-applicable band and the row for the
    currently-applicable quality classification."""
    rows = [
        ("Very good to excellent", "< 0.04", "< 0.03"),
        ("Good to fair", "0.04 – 0.07", "0.03 – 0.05"),
        ("Poor", "0.07 – 0.14", "0.05 – 0.10"),
        ("Very poor", "> 0.14", "> 0.10"),
    ]

    def cell_style(is_active_col: bool, is_active_row: bool) -> str:
        if is_active_col and is_active_row:
            return f"background:{PRIMARY}14; font-weight:700; color:{PRIMARY};"
        if is_active_col:
            return f"background:{PRIMARY}0A;"
        return ""

    col1_active = active_band == "OCR 1-2"
    col2_active = active_band == "OCR 2-4"

    header_style_base = f"padding:0.4rem 0.55rem; text-align:left; font-family:{FONT_MONO}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; color:{INK_SOFT}; border-bottom:1px solid {LINE};"
    header1 = header_style_base + (f"background:{PRIMARY}14; color:{PRIMARY}; font-weight:700;" if col1_active else "")
    header2 = header_style_base + (f"background:{PRIMARY}14; color:{PRIMARY}; font-weight:700;" if col2_active else "")

    body_rows = ""
    for label, ocr12, ocr24 in rows:
        is_row_active = (label == active_label)
        color = QUALITY_COLORS.get(label, NEUTRAL)
        row_label_style = f"padding:0.4rem 0.55rem; font-size:0.85rem; border-bottom:1px solid {LINE};"
        c1_style = f"padding:0.4rem 0.55rem; font-family:{FONT_MONO}; font-size:0.85rem; border-bottom:1px solid {LINE}; {cell_style(col1_active, is_row_active)}"
        c2_style = f"padding:0.4rem 0.55rem; font-family:{FONT_MONO}; font-size:0.85rem; border-bottom:1px solid {LINE}; {cell_style(col2_active, is_row_active)}"
        marker = f'<span class="sd-dot" style="background:{color}; margin-right:0.4rem;"></span>'
        body_rows += (
            f'<tr>'
            f'<td style="{row_label_style}">{marker}{label}</td>'
            f'<td style="{c1_style}">{ocr12}</td>'
            f'<td style="{c2_style}">{ocr24}</td>'
            f'</tr>'
        )

    return f"""
<table style="width:100%; border-collapse:collapse;">
  <thead>
    <tr>
      <th style="{header_style_base}">Sample quality</th>
      <th style="{header1}">OCR 1–2</th>
      <th style="{header2}">OCR 2–4</th>
    </tr>
  </thead>
  <tbody>
    {body_rows}
  </tbody>
</table>
"""


def percentile_position(value: float, d: dict) -> float:
    """Piecewise-linear percentile position (0-100) vs stored grid, extrapolating
    linearly beyond min/max so out-of-range values show clearly off the scale."""
    grid_x = [d["minimum"], d["p01"], d["mean"], d["p99"], d["maximum"]]
    grid_p = [0.0, 1.0, 50.0, 99.0, 100.0]
    if value <= grid_x[0]:
        if grid_x[1] == grid_x[0]:
            return grid_p[0]
        slope = (grid_p[1] - grid_p[0]) / (grid_x[1] - grid_x[0])
        return grid_p[0] + slope * (value - grid_x[0])
    if value >= grid_x[-1]:
        if grid_x[-1] == grid_x[-2]:
            return grid_p[-1]
        slope = (grid_p[-1] - grid_p[-2]) / (grid_x[-1] - grid_x[-2])
        return grid_p[-1] + slope * (value - grid_x[-1])
    return float(np.interp(value, grid_x, grid_p))


def percentile_bars_html(input_df, domain_lookup) -> str:
    rows = []
    for feature in FEATURES:
        value = float(input_df.iloc[0][feature])
        d = domain_lookup[feature]
        pct = percentile_position(value, d)
        clipped = min(max(pct, -15), 115)
        pos = (clipped + 15) / 130 * 100
        color = GOOD if 0 <= pct <= 100 else (WARN if -15 <= pct <= 115 else BAD)
        rows.append(f"""
<div style="margin-bottom:0.6rem;">
  <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:2px;">
    <span style="font-weight:600;">{SHORT_LABELS[feature]}</span>
    <span style="font-family:{FONT_MONO}; color:{INK_SOFT};">{value:g} · p{pct:.0f}</span>
  </div>
  <div style="position:relative; height:8px; border-radius:4px; background:{LINE};">
    <div style="position:absolute; left:11.5%; width:76.9%; top:0; bottom:0; background:{PAPER}; border-left:1px dashed {INK_SOFT}88; border-right:1px dashed {INK_SOFT}88;"></div>
    <div style="position:absolute; left:{pos:.1f}%; top:-2px; width:2px; height:12px; background:{color}; transform:translateX(-1px);"></div>
  </div>
</div>""")
    return "".join(rows)


# =============================================================================
# REPORT  (unchanged)
# =============================================================================
def make_excel_report(input_df, prediction, ad, shap_table, pipeline_dir, quality):
    summary = pd.DataFrame(
        {
            "Metric": [
                "Predicted SampleDisturbance (Δe/e0)",
                "Observed min-max domain",
                "Robust 1st-99th percentile domain",
                "Standardized multivariate distance",
                "Multivariate distance threshold",
                "Inside multivariate 95th-percentile domain",
                "Inside combined applicability domain",
                "Lunne sample-quality classification",
                "Lunne classification criterion",
                "Lunne OCR band applied",
                "Pipeline output directory",
            ],
            "Value": [
                prediction,
                ad["strict_inside"],
                ad["robust_inside"],
                ad["multivariate_distance"],
                ad["multivariate_threshold"],
                ad["inside_multivariate"],
                ad["inside_combined"],
                quality["label"],
                quality["criterion"],
                quality.get("band") or "n/a (OCR outside 1-4)",
                str(pipeline_dir),
            ],
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="Inputs", index=False)
        summary.to_excel(writer, sheet_name="Prediction", index=False)
        ad["feature_table"].to_excel(writer, sheet_name="Applicability", index=False)
        shap_table.drop(columns=["|SHAP|"]).to_excel(writer, sheet_name="Local_SHAP", index=False)
    return output.getvalue()


# =============================================================================
# LOAD EXACT RESEARCH OUTPUTS  (unchanged)
# =============================================================================
pipeline_dir = discover_pipeline_output_dir()

if pipeline_dir is None:
    st.markdown(
        '<div class="sd-masthead"><div class="sd-masthead-title">'
        'Soil Sampling Disturbance Predictor</div>'
        '<div class="sd-masthead-sub">MODEL UNAVAILABLE</div></div>',
        unsafe_allow_html=True,
    )
    st.error("Required research-pipeline outputs were not found.")
    st.markdown(
        "Place the following files together in a folder named `pipeline_outputs/` "
        "next to `app.py`, or run the app from the parent directory containing your "
        "latest `Results_*` folder:"
    )
    st.code(
        "best_model.joblib\n"
        "applicability_domain.xlsx\n"
        "best_model_summary.xlsx              # optional\n"
        "reproducibility_settings.xlsx        # optional\n"
        "software_versions.json               # optional\n"
        "best_model_parameters.json           # optional",
        language="text",
    )
    st.stop()

try:
    model = load_model(str(pipeline_dir / "best_model.joblib"))
    ranges_df, method_df, optional_outputs = load_pipeline_outputs(str(pipeline_dir))
    explainer = get_explainer(model)
except Exception as exc:
    st.error(f"Could not load pipeline outputs: {exc}")
    st.stop()

model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None:
    model_feature_names = list(model_feature_names)
    if model_feature_names != FEATURES:
        st.error(
            "The saved model feature order does not match the expected research model.\n\n"
            f"Model: {model_feature_names}\n\nExpected: {FEATURES}"
        )
        st.stop()

domain_lookup = build_domain_lookup(ranges_df)
multivariate_threshold = float(method_df.loc[0, "Multivariate_Distance_Threshold"])


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class="sd-masthead">
        <div class="sd-masthead-title">Soil Sampling Disturbance Predictor</div>
        <div class="sd-masthead-sub">XGBoost · Research Pipeline Output</div>
    </div>
    <div class="sd-tagline">
        Predicts sample disturbance (Δe/e₀) from sampler geometry, soil plasticity,
        and in-situ stress state, using the final model generated by the research
        pipeline. <b>Research / decision-support tool — not a substitute for
        engineering judgement.</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### Research model")
    st.markdown(pill_html("Pipeline outputs loaded", GOOD), unsafe_allow_html=True)
    st.caption(
        "Using the saved final model and applicability-domain criteria produced "
        "by the research pipeline — no fitting, tuning, or validation happens "
        "in this app."
    )
    st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">Loaded files</div>',
                unsafe_allow_html=True)
    st.markdown(f'<span class="sd-file-chip">best_model.joblib</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="sd-file-chip">applicability_domain.xlsx</span>', unsafe_allow_html=True)

    st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">Applicability criteria</div>',
                unsafe_allow_html=True)
    st.markdown("""
- Observed min–max range
- 1st–99th percentile robust range
- Standardized multivariate distance
- Combined AD = robust range **and** multivariate criterion
""")

    st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">Directory</div>',
                unsafe_allow_html=True)
    st.caption(str(pipeline_dir))


tab_predict, tab_info = st.tabs(["Predict", "Pipeline & model info"])


# =============================================================================
# TAB 1 — PREDICT
# =============================================================================
with tab_predict:
    st.markdown('<div class="sd-eyebrow">1 · Input parameters</div>', unsafe_allow_html=True)
    st.caption(
        "Fields intentionally permit values beyond the development-data range so "
        "the app can identify extrapolation."
    )

    INPUT_GROUPS = [
        ("Sampler geometry", ["AreaRatio", "CuttingEdge"]),
        ("Soil properties", ["PlasticityIndex", "OCR"]),
        ("In-situ stress state", ["VerticalStress"]),
    ]

    input_values = {}

    def render_feature_box(feature: str):
        d = domain_lookup[feature]
        default = float(d["mean"])
        span = max(d["maximum"] - d["minimum"], abs(d["maximum"]), 1.0)
        ui_min = max(0.0, d["minimum"] - 0.5 * span) if feature != "VerticalStress" else 0.0
        ui_max = d["maximum"] + 0.75 * span
        step = 0.1 if feature in {"OCR", "AreaRatio", "CuttingEdge", "PlasticityIndex"} else 1.0
        unit = FEATURE_UNITS[feature]

        with st.container(border=True):
            st.markdown(
                f'<div class="sd-field-label">{FEATURE_LABELS[feature]}'
                f'<span class="sd-field-unit">{unit}</span></div>',
                unsafe_allow_html=True,
            )
            input_values[feature] = st.number_input(
                f"{feature} value",
                min_value=float(ui_min),
                max_value=float(ui_max),
                value=default,
                step=float(step),
                format="%.3f" if feature == "OCR" else "%.2f",
                help=FEATURE_HELP[feature],
                label_visibility="collapsed",
                key=f"input_{feature}",
            )
            st.markdown(
                f'<div class="sd-input-caption">Observed {d["minimum"]:.3g}–{d["maximum"]:.3g} {unit}'
                f' &nbsp;·&nbsp; 1st–99th {d["p01"]:.3g}–{d["p99"]:.3g} {unit}</div>',
                unsafe_allow_html=True,
            )

    def render_panel(title: str, features: list[str]):
        st.markdown(f'<div class="sd-panel-title">{title}</div>', unsafe_allow_html=True)
        cols = st.columns(len(features), gap="small")
        for col, feature in zip(cols, features):
            with col:
                render_feature_box(feature)

    row1_col1, row1_col2 = st.columns(2, gap="large")
    with row1_col1:
        render_panel(*INPUT_GROUPS[0])
    with row1_col2:
        render_panel(*INPUT_GROUPS[1])

    row2_col1, row2_col2, row2_col3 = st.columns([1, 2, 1], gap="large")
    with row2_col2:
        render_panel(*INPUT_GROUPS[2])

    predict_clicked = st.button(
        "▶  PREDICT SAMPLING DISTURBANCE",
        type="primary",
        width="stretch",
    )

    if predict_clicked:
        input_df = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)

        prediction = float(model.predict(input_df)[0])
        ad = evaluate_applicability(input_df, domain_lookup, multivariate_threshold)
        explanation = explainer(input_df)
        shap_table = make_shap_table(input_df, explanation)
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        quality = classify_lunne_quality(prediction, float(input_df.iloc[0]["OCR"]))
        quality_color = QUALITY_COLORS.get(quality["label"], NEUTRAL)

        st.markdown('<div class="sd-eyebrow" style="margin-top:0.4rem;">2 · Prediction summary</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sd-card">', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Δe/e₀", f"{prediction:.4f}")
        m2.metric("Observed min–max", "INSIDE" if ad["strict_inside"] else "OUTSIDE")
        m3.metric("Combined AD", "WITHIN" if ad["inside_combined"] else "OUTSIDE")
        m4.metric(
            "Multivariate distance, D",
            f"{ad['multivariate_distance']:.3f}",
            delta=f"Threshold {ad['multivariate_threshold']:.3f}",
            delta_color="off",
        )

        q1, q2 = st.columns([0.28, 0.72])
        with q1:
            st.markdown(
                '<div style="font-weight:600; font-size:0.85rem; margin-bottom:0.4rem;">'
                'Lunne sample quality</div>', unsafe_allow_html=True,
            )
            st.markdown(pill_html(quality["label"], quality_color), unsafe_allow_html=True)
            st.caption(quality["criterion"])
        with q2:
            st.markdown(
                f'<div class="{ad["css"]}"><span class="status-label">{ad["display_level"]}</span>'
                f'{ad["message"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">3 · Applicability-domain assessment</div>',
                    unsafe_allow_html=True)
        left, right = st.columns([0.55, 0.45], gap="large")
        with left:
            st.markdown('<div class="sd-card">', unsafe_allow_html=True)
            display_ad = ad["feature_table"].copy()
            numeric_cols = ["Input", "Observed min", "1st percentile", "99th percentile", "Observed max"]
            display_ad[numeric_cols] = display_ad[numeric_cols].round(4)
            st.dataframe(display_ad, width="stretch", hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Robust 1st–99th percentile:**", "PASS" if ad["robust_inside"] else "FAIL")
            with c2:
                st.write("**Multivariate 95th-percentile:**", "PASS" if ad["inside_multivariate"] else "FAIL")
            st.caption(
                "Combined AD reproduces the research pipeline: all five predictors must "
                "lie within their 1st–99th percentile ranges and the standardized "
                "multivariate distance must be below the pipeline threshold."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="sd-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-weight:600; font-size:0.86rem; margin-bottom:0.3rem;">'
                'Inputs vs. training distribution</div>', unsafe_allow_html=True,
            )
            st.markdown(percentile_bars_html(input_df, domain_lookup), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="quality-box">
                    <div class="quality-title">Lunne et al. (1997) sample-quality criteria</div>
                    {lunne_table_html(quality.get("band"), quality["label"] if quality["applicable"] else None)}
                    <div style="font-size:0.76rem; color:{INK_SOFT}; margin-top:0.55rem;">
                        Δe = change in void ratio between initial void ratio (e₀) and
                        void ratio at in-situ stress (e<sub>in-situ</sub>). Criteria are
                        defined only for 1 ≤ OCR ≤ 4; the current input's OCR
                        ({float(input_df.iloc[0]["OCR"]):.2f}) is
                        {"within" if quality["applicable"] else "outside"} that range,
                        so the {"highlighted column above applies" if quality["applicable"] else "classification shows as Not classified"}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">4 · Local SHAP explanation</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sd-card">', unsafe_allow_html=True)

        shap_sum = float(np.asarray(explanation.values[0], dtype=float).sum())
        reconstructed = base_value + shap_sum
        s1, s2, s3 = st.columns(3)
        s1.metric("SHAP baseline", f"{base_value:.4f}")
        s2.metric("Σ SHAP contributions", format_shap_value(shap_sum))
        s3.metric("Baseline + SHAP", f"{reconstructed:.4f}")

        shap_left, shap_right = st.columns([1.15, 0.85], gap="large")
        with shap_left:
            st.markdown("**Waterfall — baseline to prediction**")
            st.plotly_chart(make_waterfall_figure(shap_table, base_value), width="stretch")
        with shap_right:
            st.markdown("**Contribution by feature**")
            st.plotly_chart(make_shap_bar_figure(shap_table), width="stretch")

        st.caption(
            "Positive SHAP values increase the prediction relative to the model "
            "baseline; negative values decrease it. SHAP explains the saved "
            "model's behaviour and does not establish causality."
        )

        with st.expander("SHAP contribution table"):
            shap_display = shap_table.drop(columns=["|SHAP|"]).copy()
            shap_display["Input"] = shap_display["Input"].map(lambda x: f"{float(x):.3f}")
            shap_display["SHAP contribution"] = shap_display["SHAP contribution"].map(format_shap_value)
            st.dataframe(shap_display, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        report_bytes = make_excel_report(input_df, prediction, ad, shap_table, pipeline_dir, quality)
        st.download_button(
            "Download prediction report (Excel)",
            data=report_bytes,
            file_name="soil_disturbance_prediction_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )


# =============================================================================
# TAB 2 — PIPELINE & MODEL INFO
# =============================================================================
with tab_info:
    st.markdown('<div class="sd-eyebrow">Loaded artifacts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-card">', unsafe_allow_html=True)
    st.write("**Loaded model:**", str(pipeline_dir / "best_model.joblib"))
    st.write("**Loaded applicability domain:**", str(pipeline_dir / "applicability_domain.xlsx"))
    st.markdown("</div>", unsafe_allow_html=True)

    best_summary = optional_outputs.get("best_model_summary.xlsx")
    if isinstance(best_summary, pd.DataFrame) and not best_summary.empty:
        st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">Best model summary</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sd-card">', unsafe_allow_html=True)
        st.dataframe(best_summary, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    settings = optional_outputs.get("reproducibility_settings.xlsx")
    if isinstance(settings, pd.DataFrame) and not settings.empty:
        st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">Reproducibility settings</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sd-card">', unsafe_allow_html=True)
        st.dataframe(settings, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    versions = optional_outputs.get("software_versions.json")
    if isinstance(versions, dict):
        with c1:
            st.markdown('<div class="sd-eyebrow">Software versions</div>', unsafe_allow_html=True)
            st.markdown('<div class="sd-card">', unsafe_allow_html=True)
            st.json(versions)
            st.markdown("</div>", unsafe_allow_html=True)

    params = optional_outputs.get("best_model_parameters.json")
    if isinstance(params, dict):
        with c2:
            st.markdown('<div class="sd-eyebrow">Saved final hyperparameters</div>', unsafe_allow_html=True)
            st.markdown('<div class="sd-card">', unsafe_allow_html=True)
            st.json(params)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sd-eyebrow" style="margin-top:0.9rem;">Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-card">', unsafe_allow_html=True)
    st.info(
        "No model fitting, tuning, or validation is performed in this app. "
        "Predictions use the final model generated by the research pipeline."
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.divider()
st.caption(
    "Research-use decision-support interface. Predictions outside the development-data "
    "applicability domain should be treated as extrapolations or weakly supported estimates "
    "and interpreted with appropriate engineering judgement."
)
