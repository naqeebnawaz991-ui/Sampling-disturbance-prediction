from __future__ import annotations

from pathlib import Path
from datetime import datetime
import io
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Soil Sampling Disturbance Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

FEATURES = [
    "AreaRatio",
    "CuttingEdge",
    "PlasticityIndex",
    "OCR",
    "VerticalStress",
]

FEATURE_LABELS = {
    "AreaRatio": "Area ratio, AR (%)",
    "CuttingEdge": "Cutting-edge angle, CE (°)",
    "PlasticityIndex": "Plasticity index, PI (%)",
    "OCR": "Overconsolidation ratio, OCR",
    "VerticalStress": "In-situ vertical effective stress, σ′v (kPa)",
}

SHORT_LABELS = {
    "AreaRatio": "AR (%)",
    "CuttingEdge": "CE (°)",
    "PlasticityIndex": "PI (%)",
    "OCR": "OCR",
    "VerticalStress": "σ′v (kPa)",
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
# STYLING
# =============================================================================
st.markdown(
    """
    <style>
    :root{
        --navy:#092d63;
        --navy2:#123f7a;
        --blue:#164ea5;
        --line:#b8c7d9;
        --soft:#f6f9fd;
        --softblue:#f3f8ff;
        --muted:#667085;
        --green:#147a2e;
        --greenbg:#eef9f0;
        --gold:#a56500;
        --goldbg:#fff8e9;
        --purple:#6335b5;
        --purplebg:#f7f2ff;
        --orange:#c46b00;
        --orangebg:#fff7ec;
    }

    .block-container{
        max-width:820px;
        padding-top:.65rem;
        padding-bottom:1.8rem;
    }

    h1{
        color:var(--navy);
        text-align:left;
        font-size:2.05rem;
        font-weight:820;
        letter-spacing:-.02em;
        margin-bottom:.08rem;
    }

    .hero-subtitle{
        color:#344054;
        font-size:.86rem;
        margin-bottom:.9rem;
    }

    .app-header{
        border:2px solid var(--navy);
        border-radius:14px;
        background:#fff;
        padding:1rem 1.1rem .85rem;
        margin-bottom:1rem;
        box-shadow:0 2px 0 rgba(9,45,99,.08);
    }

    .section-shell,
    .result-shell,
    .history-shell{
        border:2px solid #c7d2df;
        border-radius:14px;
        background:#fff;
        padding:.95rem 1rem 1rem;
        margin:.8rem 0;
        position:relative;
        box-shadow:0 1px 2px rgba(15,23,42,.04);
    }

    .section-head{
        display:flex;
        align-items:center;
        gap:.5rem;
        color:var(--navy);
        font-weight:800;
        font-size:1.12rem;
        margin-bottom:.75rem;
    }

    .step-badge{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:2.15rem;
        height:2.15rem;
        border-radius:8px;
        background:var(--navy);
        color:#fff;
        font-size:1rem;
        font-weight:850;
        box-shadow:0 2px 0 rgba(9,45,99,.18);
    }

    .section-icon{
        font-size:1.35rem;
        line-height:1;
    }

    .input-note{
        color:var(--muted);
        font-size:.78rem;
        margin:-.15rem 0 .55rem;
    }

    /* Same style/color for every input */
    .input-wrap{
        border:2px solid #b8cae1;
        border-radius:11px;
        background:var(--softblue);
        padding:.55rem .65rem .48rem;
        margin-bottom:.35rem;
    }

    div[data-testid="stNumberInput"] label p{
        color:var(--navy)!important;
        font-size:.96rem!important;
        font-weight:760!important;
    }

    div[data-testid="stNumberInput"] input{
        background:#fff!important;
        border:1.8px solid #aebfd3!important;
        min-height:2.7rem!important;
        font-size:1.04rem!important;
        font-weight:760!important;
        color:#111827!important;
    }

    div[data-testid="stNumberInput"] button{
        border-color:#aebfd3!important;
        background:#f7f9fc!important;
    }

    .input-caption{
        color:#385b8e;
        font-size:.72rem;
        line-height:1.25;
        margin-top:.08rem;
    }

    div.stButton > button[kind="primary"]{
        min-height:3rem;
        border-radius:9px;
        border:2px solid #072653;
        background:var(--navy);
        color:#fff;
        font-weight:800;
        font-size:.95rem;
        box-shadow:0 2px 0 rgba(9,45,99,.14);
    }

    div.stButton > button[kind="primary"]:hover{
        background:var(--navy2);
        border-color:#072653;
    }

    .success-line{
        border:1.8px solid #a8d8b1;
        background:#effaf1;
        color:#176b2b;
        border-radius:9px;
        padding:.7rem .8rem;
        font-size:.88rem;
        margin-bottom:.75rem;
    }

    .result-two{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:.75rem;
        margin-bottom:.75rem;
    }

    .prediction-box{
        border:2px solid #acd0ff;
        border-radius:12px;
        background:linear-gradient(180deg,#f5f9ff 0%,#fff 100%);
        padding:.9rem;
        text-align:center;
    }

    .quality-box-main{
        border:2px solid #efc16e;
        border-radius:12px;
        background:linear-gradient(180deg,#fffaf0 0%,#fff 100%);
        padding:.9rem;
        text-align:center;
    }

    .box-label{
        color:var(--navy);
        font-size:.82rem;
        font-weight:720;
        margin-bottom:.15rem;
    }

    .pred-value{
        color:#0a44a2;
        font-size:2.65rem;
        font-weight:850;
        line-height:1.05;
        margin:.18rem 0;
    }

    .quality-pill{
        display:inline-block;
        border:1.7px solid currentColor;
        border-radius:999px;
        padding:.3rem .8rem;
        font-weight:820;
        font-size:1.02rem;
        background:#fff;
        margin:.2rem 0;
    }

    .quality-excellent{color:#2e7d32;}
    .quality-goodfair{color:#a56500;}
    .quality-poor{color:#c46b00;}
    .quality-verypoor{color:#b42318;}

    .result-grid{
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:.55rem;
    }

    .mini-card{
        border:2px solid #b8c7d9;
        border-radius:10px;
        background:#fff;
        padding:.65rem .55rem;
        text-align:center;
        min-height:95px;
    }

    .mini-card.blue{border-color:#a9c8ed;background:#f6faff;}
    .mini-card.green{border-color:#a8d3ad;background:#f7fcf8;}
    .mini-card.purple{border-color:#c7b0e9;background:#faf7ff;}
    .mini-card.orange{border-color:#efc48b;background:#fffaf3;}

    .mini-label{
        font-size:.72rem;
        font-weight:760;
        margin-bottom:.18rem;
    }

    .mini-value{
        font-size:1.25rem;
        font-weight:850;
        color:var(--navy);
    }

    .mini-sub{
        color:var(--muted);
        font-size:.68rem;
        margin-top:.15rem;
        line-height:1.2;
    }

    .status-good,.status-warn,.status-bad{
        border-radius:9px;
        padding:.65rem .75rem;
        margin-top:.55rem;
        font-size:.78rem;
        line-height:1.35;
    }
    .status-good{border:1.8px solid #9fd0a8;background:#eff8f1;color:#226a31;}
    .status-warn{border:1.8px solid #e4b35b;background:#fff8e9;color:#805000;}
    .status-bad{border:1.8px solid #dc8a82;background:#fff2f1;color:#8d1c13;}

    div[data-testid="stDataFrame"]{
        border:1.8px solid #c8d3df;
        border-radius:9px;
        overflow:hidden;
    }

    .technical-head{
        color:var(--navy);
        font-weight:800;
        font-size:1rem;
        margin:.3rem 0 .45rem;
    }

    .criterion-row{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:.5rem;
        margin:.4rem 0;
    }

    .criterion-box{
        border:1.8px solid #b8c7d9;
        border-radius:9px;
        background:#f7f9fc;
        padding:.5rem;
        text-align:center;
        color:#35516f;
        font-size:.75rem;
        font-weight:700;
    }

    .threshold-box{
        border:1.8px solid #e2c789;
        background:#fffaf0;
        border-radius:9px;
        padding:.6rem .7rem;
        font-size:.76rem;
        line-height:1.45;
        margin-top:.5rem;
    }

    .shap-meta{
        border:1.8px solid #c9d6e4;
        border-radius:9px;
        background:#f8fafc;
        padding:.5rem .65rem;
        text-align:center;
        font-size:.75rem;
        color:#35516f;
        margin-bottom:.45rem;
    }

    [data-testid="stSidebar"]{
        background:#f5f8fc;
        border-right:1.8px solid #b9c8d7;
    }

    .footer-note{
        color:var(--muted);
        text-align:center;
        font-size:.71rem;
        margin-top:.7rem;
    }

    @media(max-width:700px){
        .block-container{max-width:96vw;}
        .result-two,.result-grid,.criterion-row{grid-template-columns:1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PIPELINE OUTPUT LOADING
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
# APPLICABILITY DOMAIN
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
# LUNNE CLASSIFICATION AND SHAP HELPERS
# =============================================================================
def classify_lunne_quality(prediction: float, ocr: float):
    """
    Classify predicted sample disturbance using the four Lunne quality classes
    requested for the GUI. The same Δe/e₀ thresholds are displayed for all OCR values.
    """
    if prediction < 0.04:
        return {
            "label": "Excellent",
            "css": "quality-excellent",
            "criterion": "Δe/e₀ < 0.04",
            "applicable": True,
        }
    if prediction < 0.07:
        return {
            "label": "Good–Fair",
            "css": "quality-goodfair",
            "criterion": "0.04 ≤ Δe/e₀ < 0.07",
            "applicable": True,
        }
    if prediction < 0.14:
        return {
            "label": "Poor",
            "css": "quality-poor",
            "criterion": "0.07 ≤ Δe/e₀ < 0.14",
            "applicable": True,
        }
    return {
        "label": "Very Poor",
        "css": "quality-verypoor",
        "criterion": "Δe/e₀ ≥ 0.14",
        "applicable": True,
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


def make_display_explanation(explanation):
    return shap.Explanation(
        values=np.asarray(explanation.values[0], dtype=float),
        base_values=float(np.asarray(explanation.base_values).reshape(-1)[0]),
        data=np.asarray(explanation.data[0], dtype=float),
        feature_names=[SHORT_LABELS[f] for f in FEATURES],
    )


def make_waterfall(explanation):
    display_explanation = make_display_explanation(explanation)
    plt.figure(figsize=(6.6, 4.1))
    shap.plots.waterfall(
        display_explanation,
        max_display=len(FEATURES),
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# =============================================================================
# REPORT
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
                str(pipeline_dir),
            ],
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="Inputs", index=False)
        summary.to_excel(writer, sheet_name="Prediction", index=False)
        ad["feature_table"].to_excel(
            writer, sheet_name="Applicability", index=False
        )
        shap_table.drop(columns=["|SHAP|"]).to_excel(
            writer, sheet_name="Local_SHAP", index=False
        )
    return output.getvalue()


# =============================================================================
# LOAD EXACT RESEARCH OUTPUTS
# =============================================================================
pipeline_dir = discover_pipeline_output_dir()

if pipeline_dir is None:
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
multivariate_threshold = float(
    method_df.loc[0, "Multivariate_Distance_Threshold"]
)


# =============================================================================
# APP HEADER AND SIDEBAR
# =============================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>📊 Soil Sampling Disturbance Prediction</h1>
        <div class="hero-subtitle">
            Enter sampler geometry and soil parameters to predict Δe/e₀ using the final XGBoost model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Research model")
    st.success("Research outputs loaded")
    st.caption("Saved final XGBoost model and applicability-domain outputs are being used.")
    st.markdown("**Model files**")
    st.code("best_model.joblib\napplicability_domain.xlsx", language="text")
    st.markdown("**Applicability domain**")
    st.caption("Observed range • 1st–99th percentile range • standardized multivariate distance")
    st.info("No fitting, tuning, or validation is performed in this interface.")


# =============================================================================
# INPUTS
# =============================================================================
st.markdown(
    '<div class="section-shell">'
    '<div class="section-head"><span class="step-badge">1</span>'
    '<span class="section-icon">⚙️</span>Input Parameters for Prediction</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="input-note">Enter the five model inputs. Values outside the development range are allowed and will be flagged.</div>',
    unsafe_allow_html=True,
)

input_values = {}

def render_input(feature, col):
    d = domain_lookup[feature]
    default = float(d["mean"])
    span = max(d["maximum"] - d["minimum"], abs(d["maximum"]), 1.0)
    ui_min = max(0.0, d["minimum"] - 0.5 * span) if feature != "VerticalStress" else 0.0
    ui_max = d["maximum"] + 0.75 * span
    step = 0.1 if feature in {"OCR", "AreaRatio", "CuttingEdge", "PlasticityIndex"} else 1.0

    with col:
        st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
        input_values[feature] = st.number_input(
            FEATURE_LABELS[feature],
            min_value=float(ui_min),
            max_value=float(ui_max),
            value=float(default),
            step=float(step),
            format="%.3f" if feature == "OCR" else "%.2f",
        )
        st.markdown(
            f'<div class="input-caption">'
            f'Observed: {d["minimum"]:.3g}–{d["maximum"]:.3g} &nbsp; | &nbsp; '
            f'1st–99th: {d["p01"]:.3g}–{d["p99"]:.3g}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

r1 = st.columns(2, gap="medium")
render_input(FEATURES[0], r1[0])
render_input(FEATURES[1], r1[1])

r2 = st.columns(2, gap="medium")
render_input(FEATURES[2], r2[0])
render_input(FEATURES[3], r2[1])

r3 = st.columns([0.12, 0.76, 0.12])
render_input(FEATURES[4], r3[1])

predict_clicked = st.button(
    "🧠  Predict Sampling Disturbance",
    type="primary",
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =============================================================================
# RESULTS
# =============================================================================
if predict_clicked:
    input_df = pd.DataFrame(
        [[input_values[f] for f in FEATURES]],
        columns=FEATURES,
    )

    prediction = float(model.predict(input_df)[0])
    ad = evaluate_applicability(
        input_df,
        domain_lookup,
        multivariate_threshold,
    )
    explanation = explainer(input_df)
    shap_table = make_shap_table(input_df, explanation)
    quality = classify_lunne_quality(
        prediction,
        float(input_df.iloc[0]["OCR"]),
    )
    st.markdown(
        '<div class="result-shell">'
        '<div class="section-head"><span class="step-badge">2</span>'
        '<span class="section-icon">🔮</span>Prediction Result</div>',
        unsafe_allow_html=True,
    )

    quality_css = quality["css"]
    observed_text = "INSIDE" if ad["strict_inside"] else "OUTSIDE"
    combined_text = "WITHIN" if ad["inside_combined"] else "OUTSIDE"
    ad_text = "WITHIN" if ad["inside_multivariate"] else "OUTSIDE"

    st.markdown(
        f'<div class="success-line">✅ Prediction complete! '
        f'Δe/e₀ value: <b>{prediction:.4f}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-two">
            <div class="prediction-box">
                <div class="box-label">Predicted Δe/e₀</div>
                <div class="pred-value">{prediction:.4f}</div>
            </div>
            <div class="quality-box-main">
                <div class="box-label">Sample Quality Category</div>
                <div class="{quality_css} quality-pill">{quality["label"]}</div>
                <div style="font-size:.72rem;color:#667085;margin-top:.12rem;">{quality["criterion"]}</div>
            </div>
        </div>

        <div class="result-grid">
            <div class="mini-card blue">
                <div class="mini-label" style="color:#174da1;">Observed Range</div>
                <div class="mini-value">{observed_text}</div>
                <div class="mini-sub">All inputs within observed min–max.</div>
            </div>
            <div class="mini-card green">
                <div class="mini-label" style="color:#2f7d3d;">Combined AD</div>
                <div class="mini-value">{combined_text}</div>
                <div class="mini-sub">Robust range + multivariate criterion.</div>
            </div>
            <div class="mini-card purple">
                <div class="mini-label" style="color:#6335b5;">Multivariate Distance, D</div>
                <div class="mini-value">{ad["multivariate_distance"]:.3f}</div>
                <div class="mini-sub">Threshold = {ad["multivariate_threshold"]:.3f}</div>
            </div>
            <div class="mini-card orange">
                <div class="mini-label" style="color:#c46b00;">AD Criterion</div>
                <div class="mini-value">{ad_text}</div>
                <div class="mini-sub">D compared with saved threshold.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not ad["inside_combined"] or not ad["strict_inside"]:
        st.markdown(
            f'<div class="{ad["css"]}"><b>{ad["display_level"]}</b><br>{ad["message"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Append current prediction to history.
    history_row = {
        "AR (%)": float(input_df.iloc[0]["AreaRatio"]),
        "CE (°)": float(input_df.iloc[0]["CuttingEdge"]),
        "PI (%)": float(input_df.iloc[0]["PlasticityIndex"]),
        "OCR": float(input_df.iloc[0]["OCR"]),
        "σ′v (kPa)": float(input_df.iloc[0]["VerticalStress"]),
        "Predicted Δe/e₀": prediction,
        "Category": quality["label"],
        "Date & Time": datetime.now().strftime("%d %b %Y %H:%M"),
    }
    st.session_state.prediction_history.append(history_row)
    # ---------------------------------------------------------------------
    # STEP 3: Prediction history
    # ---------------------------------------------------------------------
    st.markdown(
        '<div class="history-shell">'
        '<div class="section-head"><span class="step-badge">3</span>'
        '<span class="section-icon">📋</span>Prediction History</div>',
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_display = history_df.copy()
    if not history_display.empty:
        for col in ["AR (%)", "CE (°)", "PI (%)", "OCR", "σ′v (kPa)", "Predicted Δe/e₀"]:
            history_display[col] = pd.to_numeric(history_display[col], errors="coerce").round(4)

    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True,
        height=min(255, 70 + 35 * max(len(history_display), 1)),
    )

    history_csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Download History (CSV)",
        data=history_csv,
        file_name="soil_disturbance_prediction_history.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # Technical detail: applicability-domain assessment
    # ---------------------------------------------------------------------
    with st.expander("Applicability-domain assessment"):
        display_ad = ad["feature_table"].copy()
        display_ad["Input"] = display_ad["Input"].round(3)
        display_ad["Observed"] = display_ad.apply(
            lambda r: f'{r["Observed min"]:.3g}–{r["Observed max"]:.3g}', axis=1
        )
        display_ad["1st–99th"] = display_ad.apply(
            lambda r: f'{r["1st percentile"]:.3g}–{r["99th percentile"]:.3g}', axis=1
        )
        display_ad = display_ad[["Parameter","Input","Observed","1st–99th","Status"]]

        robust_text = "PASS" if ad["robust_inside"] else "FAIL"
        multi_text = "PASS" if ad["inside_multivariate"] else "FAIL"

        st.markdown(
            f'<div class="criterion-row">'
            f'<div class="criterion-box">1st–99th percentile criterion: {robust_text}</div>'
            f'<div class="criterion-box">Multivariate-distance criterion: {multi_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(display_ad, use_container_width=True, hide_index=True, height=220)

        st.markdown(
            """
            <div class="threshold-box">
                <b>Sample-quality thresholds</b><br>
                <span style="color:#2e7d32;"><b>Excellent:</b></span> Δe/e₀ &lt; 0.04 &nbsp;&nbsp;
                <span style="color:#7f6500;"><b>Good–Fair:</b></span> 0.04 ≤ Δe/e₀ &lt; 0.07<br>
                <span style="color:#c46b00;"><b>Poor:</b></span> 0.07 ≤ Δe/e₀ &lt; 0.14 &nbsp;&nbsp;
                <span style="color:#b42318;"><b>Very Poor:</b></span> Δe/e₀ ≥ 0.14
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------------
    # Technical detail: local SHAP
    # ---------------------------------------------------------------------
    with st.expander("Local SHAP explanation"):
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        shap_sum = float(np.asarray(explanation.values[0], dtype=float).sum())
        reconstructed = base_value + shap_sum

        st.markdown(
            f'<div class="shap-meta">'
            f'Baseline = {base_value:.4f} &nbsp;•&nbsp; '
            f'Σ SHAP = {format_shap_value(shap_sum)} &nbsp;•&nbsp; '
            f'Baseline + SHAP = {reconstructed:.4f}'
            f'</div>',
            unsafe_allow_html=True,
        )

        fig = make_waterfall(explanation)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        shap_display = shap_table.drop(columns=["|SHAP|"]).copy()
        shap_display["Input"] = shap_display["Input"].map(lambda x: f"{float(x):.3f}")
        shap_display["SHAP contribution"] = shap_display["SHAP contribution"].map(format_shap_value)
        st.dataframe(shap_display, use_container_width=True, hide_index=True, height=220)
        st.caption("Positive SHAP values increase the prediction; negative values decrease it.")


    with st.expander("Model and pipeline details"):
        st.write(
            "**Loaded model:**",
            str(pipeline_dir / "best_model.joblib"),
        )
        st.write(
            "**Loaded applicability domain:**",
            str(pipeline_dir / "applicability_domain.xlsx"),
        )

        best_summary = optional_outputs.get("best_model_summary.xlsx")
        if isinstance(best_summary, pd.DataFrame) and not best_summary.empty:
            st.write("**Best model summary**")
            st.dataframe(
                best_summary,
                use_container_width=True,
                hide_index=True,
            )

        settings = optional_outputs.get("reproducibility_settings.xlsx")
        if isinstance(settings, pd.DataFrame) and not settings.empty:
            st.write("**Reproducibility settings**")
            st.dataframe(
                settings,
                use_container_width=True,
                hide_index=True,
            )

        versions = optional_outputs.get("software_versions.json")
        if isinstance(versions, dict):
            st.write("**Software versions used by research pipeline**")
            st.json(versions)

        params = optional_outputs.get("best_model_parameters.json")
        if isinstance(params, dict):
            st.write("**Saved final hyperparameters**")
            st.json(params)

    report_bytes = make_excel_report(
        input_df,
        prediction,
        ad,
        shap_table,
        pipeline_dir,
        quality,
    )
    st.download_button(
        "Download prediction report (Excel)",
        data=report_bytes,
        file_name="soil_disturbance_prediction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
    )


st.markdown('<div class="footer-note">Research-use decision-support interface. Outside-domain predictions should be interpreted cautiously.</div>',unsafe_allow_html=True)
