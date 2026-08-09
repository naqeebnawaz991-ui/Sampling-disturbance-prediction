from __future__ import annotations

from pathlib import Path
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
    :root {--ink:#17365d;--line:#294f79;--soft:#b8c8d8;--muted:#667085;--bluefill:#eef6ff;--greenfill:#eef9f0;--purplefill:#f7f0ff;--orangefill:#fff6eb;--redfill:#fff0f0;}
    .block-container{max-width:790px;padding-top:.8rem;padding-bottom:1.6rem;}
    h1{color:var(--ink);text-align:center;font-size:2rem;font-weight:780;letter-spacing:-.02em;margin-bottom:.1rem;}
    h2,h3{color:var(--ink);font-weight:740;}
    .hero-subtitle{text-align:center;color:var(--muted);font-size:.86rem;margin-bottom:.8rem;}
    .section-shell{border:4px solid var(--line);border-radius:16px;background:#fff;padding:.95rem 1rem 1rem;margin:.75rem 0;box-shadow:0 3px 0 rgba(23,54,93,.10);}
    .section-head{display:flex;align-items:center;gap:.55rem;color:var(--ink);font-weight:820;font-size:1.12rem;padding-bottom:.55rem;margin-bottom:.55rem;border-bottom:3px solid #d5e0eb;}
    .step-dot{display:inline-flex;align-items:center;justify-content:center;width:1.8rem;height:1.8rem;border-radius:50%;background:var(--ink);color:#fff;font-size:.88rem;font-weight:850;}
    .input-caption{color:var(--muted);font-size:.74rem;line-height:1.2;margin-top:.04rem;}
    div[data-testid="stNumberInput"] label p{
        font-size:.98rem!important;
        font-weight:760!important;
        color:#17365d!important;
    }
    div[data-testid="stNumberInput"] input{
        font-size:1.05rem!important;
        font-weight:760!important;
        min-height:2.75rem!important;
    }
    .input-card{
        border:3px solid #b7c6d6;
        border-radius:12px;
        padding:.60rem .68rem .48rem;
        background:#f8fbff;
        margin-bottom:.35rem;
    }
    .input-card-blue{background:var(--bluefill);border-color:#9fc3ef;}
    .input-card-green{background:var(--greenfill);border-color:#a7d7ad;}
    .input-card-purple{background:var(--purplefill);border-color:#cbb3ef;}
    .input-card-orange{background:var(--orangefill);border-color:#edc28e;}
    .input-card-red{background:var(--redfill);border-color:#efb1b1;}
    .step1-note{
        color:#667085;
        font-size:.78rem;
        margin:-.05rem 0 .45rem 0;
    }
    .step2-shell{
        border:4px solid #2f7d3d;
        border-radius:16px;
        background:linear-gradient(180deg,#f6fcf7 0%,#ffffff 100%);
        padding:.95rem 1rem 1rem;
        margin:.75rem 0;
        box-shadow:0 3px 0 rgba(47,125,61,.10);
    }
    .step2-head{
        display:flex;
        align-items:center;
        gap:.55rem;
        color:#1f6a2d;
        font-weight:820;
        font-size:1.12rem;
        padding-bottom:.55rem;
        margin-bottom:.65rem;
        border-bottom:3px solid #cfe6d3;
    }
    .step2-dot{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:1.8rem;
        height:1.8rem;
        border-radius:50%;
        background:#1f6a2d;
        color:white;
        font-size:.88rem;
        font-weight:850;
    }

    div[data-testid="stNumberInput"] input{background:#fbfcfe;border:2px solid #b7c6d6!important;font-weight:650;}
    div[data-testid="stNumberInput"] button{border-color:#b7c6d6!important;}
    div.stButton>button[kind="primary"]{min-height:3.15rem;border-radius:10px;border:3px solid #17365d;font-weight:800;font-size:.96rem;box-shadow:0 2px 0 rgba(23,54,93,.16);}
    .prediction-hero{border:3px solid #9bc8a4;border-radius:14px;background:linear-gradient(180deg,#f8fdf9 0%,#fff 100%);padding:1rem;text-align:center;}
    .prediction-label{color:var(--muted);font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
    .prediction-value{color:var(--ink);font-size:2.25rem;line-height:1.05;font-weight:820;margin:.18rem 0 .25rem;}
    .quality-pill{display:inline-block;border:2px solid currentColor;border-radius:999px;padding:.28rem .7rem;font-weight:780;font-size:.9rem;background:#fff;}
    .quality-excellent{color:#2e7d32}.quality-goodfair{color:#5b7f2b}.quality-poor{color:#bf7600}.quality-verypoor{color:#b42318}
    .result-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-top:.55rem;}
    .mini-card{border:2px solid var(--soft);border-radius:10px;background:#fff;padding:.58rem .62rem;text-align:center;}
    .mini-label{color:var(--muted);font-size:.69rem;font-weight:700}.mini-value{color:var(--ink);font-size:1.02rem;font-weight:800;margin-top:.08rem}.mini-sub{color:var(--muted);font-size:.67rem;margin-top:.06rem}
    .status-good,.status-warn,.status-bad{border-radius:10px;padding:.62rem .72rem;margin-top:.5rem;font-size:.79rem;line-height:1.35}.status-good{border:2px solid #6cae75;background:#f0f8f1;color:#245c2b}.status-warn{border:2px solid #dca340;background:#fff8e9;color:#7f5100}.status-bad{border:2px solid #dc786f;background:#fff2f1;color:#8d1c13}
    .criterion-row{display:grid;grid-template-columns:1fr 1fr;gap:.48rem;margin-bottom:.5rem}.criterion-box{border:2px solid var(--soft);border-radius:10px;background:#f5f9fd;padding:.52rem .6rem;font-size:.76rem;color:#35516f;text-align:center;font-weight:700}
    .quality-box{border:2px solid #ddc78d;background:#fffaf0;border-radius:10px;padding:.62rem .72rem;font-size:.77rem;line-height:1.45;margin-top:.55rem}.quality-title{color:var(--ink);font-weight:780;margin-bottom:.22rem}
    .shap-meta{border:2px solid #ccd8e5;border-radius:10px;background:#f8fafc;padding:.46rem .62rem;font-size:.74rem;color:#35516f;margin-bottom:.45rem;text-align:center}
    div[data-testid="stDataFrame"]{border:2px solid #aebfd0;border-radius:10px;overflow:hidden;}
    [data-testid="stSidebar"]{background:#f3f7fb;border-right:2px solid #9db2c7;}
    .footer-note{color:var(--muted);text-align:center;font-size:.71rem;margin-top:.65rem;}
    @media(max-width:700px){.block-container{max-width:96vw}.result-grid,.criterion-row{grid-template-columns:1fr}}
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
st.title("Soil Sampling Disturbance Prediction")
st.markdown('<div class="hero-subtitle">Final XGBoost deployment model with applicability-domain and local SHAP interpretation</div>', unsafe_allow_html=True)

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
    '<div class="section-head"><span class="step-dot">1</span>INPUT PARAMETERS</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="step1-note">Enter the five model inputs. Values outside the development range are allowed so extrapolation can be flagged.</div>',
    unsafe_allow_html=True,
)

input_values = {}

CARD_CLASSES = {
    "AreaRatio": "input-card-blue",
    "CuttingEdge": "input-card-green",
    "PlasticityIndex": "input-card-purple",
    "OCR": "input-card-orange",
    "VerticalStress": "input-card-red",
}

def render_input(feature, col):
    d = domain_lookup[feature]
    default = float(d["mean"])
    span = max(d["maximum"] - d["minimum"], abs(d["maximum"]), 1.0)
    ui_min = max(0.0, d["minimum"] - 0.5 * span) if feature != "VerticalStress" else 0.0
    ui_max = d["maximum"] + 0.75 * span
    step = 0.1 if feature in {"OCR", "AreaRatio", "CuttingEdge", "PlasticityIndex"} else 1.0

    with col:
        st.markdown(f'<div class="input-card {CARD_CLASSES[feature]}">', unsafe_allow_html=True)
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

r1 = st.columns(2, gap="small")
render_input(FEATURES[0], r1[0])
render_input(FEATURES[1], r1[1])

r2 = st.columns(2, gap="small")
render_input(FEATURES[2], r2[0])
render_input(FEATURES[3], r2[1])

r3 = st.columns([0.08, 0.84, 0.08])
render_input(FEATURES[4], r3[1])

predict_clicked = st.button(
    "▶  PREDICT SAMPLING DISTURBANCE",
    type="primary",
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown('<div class="step2-shell"><div class="step2-head"><span class="step2-dot">2</span>PREDICTION SUMMARY</div>', unsafe_allow_html=True)
    quality_css=quality["css"]
    observed_text="Inside" if ad["strict_inside"] else "Outside"
    combined_text="Within" if ad["inside_combined"] else "Outside"
    st.markdown(f"""<div class="prediction-hero"><div class="prediction-label">Predicted sample disturbance</div><div class="prediction-value">Δe/e₀ = {prediction:.4f}</div><div class="{quality_css} quality-pill">{quality["label"]}</div><div style="color:#667085;font-size:.74rem;margin-top:.25rem;">{quality["criterion"]}</div><div class="result-grid"><div class="mini-card"><div class="mini-label">Observed range</div><div class="mini-value">{observed_text}</div></div><div class="mini-card"><div class="mini-label">Combined AD</div><div class="mini-value">{combined_text}</div></div><div class="mini-card"><div class="mini-label">Multivariate distance</div><div class="mini-value">{ad["multivariate_distance"]:.3f}</div><div class="mini-sub">Threshold {ad["multivariate_threshold"]:.3f}</div></div></div></div>""",unsafe_allow_html=True)
    if not ad["inside_combined"] or not ad["strict_inside"]:
        st.markdown(f'<div class="{ad["css"]}"><b>{ad["display_level"]}</b><br>{ad["message"]}</div>',unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="section-shell"><div class="section-head"><span class="step-dot">3</span>Applicability-domain assessment</div>', unsafe_allow_html=True)
    display_ad=ad["feature_table"].copy()
    display_ad["Input"]=display_ad["Input"].round(3)
    display_ad["Observed"]=display_ad.apply(lambda r:f'{r["Observed min"]:.3g}–{r["Observed max"]:.3g}',axis=1)
    display_ad["1st–99th"]=display_ad.apply(lambda r:f'{r["1st percentile"]:.3g}–{r["99th percentile"]:.3g}',axis=1)
    display_ad=display_ad[["Parameter","Input","Observed","1st–99th","Status"]]
    robust_text="PASS" if ad["robust_inside"] else "FAIL"
    multi_text="PASS" if ad["inside_multivariate"] else "FAIL"
    st.markdown(f'<div class="criterion-row"><div class="criterion-box">1st–99th percentile criterion: {robust_text}</div><div class="criterion-box">Multivariate-distance criterion: {multi_text}</div></div>',unsafe_allow_html=True)
    st.dataframe(display_ad,use_container_width=True,hide_index=True,height=220)
    st.markdown("""<div class="quality-box"><div class="quality-title">Sample-quality thresholds</div><b style="color:#2e7d32;">Excellent:</b> Δe/e₀ &lt; 0.04 &nbsp;&nbsp; <b style="color:#5b7f2b;">Good–Fair:</b> 0.04 ≤ Δe/e₀ &lt; 0.07<br><b style="color:#bf7600;">Poor:</b> 0.07 ≤ Δe/e₀ &lt; 0.14 &nbsp;&nbsp; <b style="color:#b42318;">Very Poor:</b> Δe/e₀ ≥ 0.14</div>""",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="section-shell"><div class="section-head"><span class="step-dot">4</span>Local SHAP explanation</div>', unsafe_allow_html=True)

    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    shap_sum = float(np.asarray(explanation.values[0], dtype=float).sum())
    reconstructed = base_value + shap_sum
    st.markdown(f'<div class="shap-meta">Baseline = {base_value:.4f} &nbsp;•&nbsp; Σ SHAP = {format_shap_value(shap_sum)} &nbsp;•&nbsp; Baseline + SHAP = {reconstructed:.4f}</div>',unsafe_allow_html=True)
    st.markdown("**SHAP waterfall plot**")
    fig=make_waterfall(explanation)
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)
    st.markdown("**Local SHAP contributions**")
    shap_display=shap_table.drop(columns=["|SHAP|"]).copy()
    shap_display["Input"]=shap_display["Input"].map(lambda x:f"{float(x):.3f}")
    shap_display["SHAP contribution"]=shap_display["SHAP contribution"].map(format_shap_value)
    st.dataframe(shap_display,use_container_width=True,hide_index=True,height=220)
    st.caption("Positive SHAP values increase the prediction; negative values decrease it.")
    st.markdown("</div>",unsafe_allow_html=True)

    with st.expander("Model details"):
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
