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
    layout="wide",
)

FEATURES = [
    "AreaRatio",
    "CuttingEdge",
    "PlasticityIndex",
    "OCR",
    "VerticalStress",
]

FEATURE_LABELS = {
    "AreaRatio": "Area ratio, AR [%]",
    "CuttingEdge": "Cutting-edge angle, CE [°]",
    "PlasticityIndex": "Plasticity index, PI [%]",
    "OCR": "Overconsolidation ratio, OCR",
    "VerticalStress": "In-situ vertical effective stress, σ′ᵥ [kPa]",
}

SHORT_LABELS = {
    "AreaRatio": "AR [%]",
    "CuttingEdge": "CE [°]",
    "PlasticityIndex": "PI [%]",
    "OCR": "OCR",
    "VerticalStress": "σ′ᵥ [kPa]",
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
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    h1 {
        color: #17365d;
        font-weight: 760;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    h2, h3 {
        color: #17365d;
        font-weight: 700;
    }

    .section-label {
        color: #17365d;
        font-size: 1.12rem;
        font-weight: 760;
        margin: 0.2rem 0 0.65rem 0;
    }

    .status-good,
    .status-warn,
    .status-bad {
        padding: 0.95rem 1.05rem;
        border-radius: 10px;
        margin: 0.65rem 0 0.4rem 0;
    }

    .status-good {
        border-left: 5px solid #2e7d32;
        background: #edf8ef;
    }

    .status-warn {
        border-left: 5px solid #d68a00;
        background: #fff8e7;
    }

    .status-bad {
        border-left: 5px solid #c62828;
        background: #fff0f0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #dfe4ea;
        padding: 0.9rem 0.95rem;
        border-radius: 12px;
        background: #ffffff;
        min-height: 116px;
    }

    div[data-testid="stMetricLabel"] {
        color: #344054;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #17365d;
        font-weight: 760;
    }

    div[data-testid="stNumberInput"] input {
        background: #f8fafc;
    }

    div.stButton > button[kind="primary"] {
        min-height: 3.15rem;
        border-radius: 8px;
        font-weight: 760;
        font-size: 1rem;
    }

    .quality-box {
        border: 1px solid #e2e8f0;
        background: #fffdf8;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-top: 0.45rem;
    }

    .quality-title {
        color: #17365d;
        font-weight: 760;
        margin-bottom: 0.35rem;
    }

    .quality-excellent {color:#2e7d32; font-weight:760;}
    .quality-goodfair {color:#558b2f; font-weight:760;}
    .quality-poor {color:#d18b00; font-weight:760;}
    .quality-verypoor {color:#c62828; font-weight:760;}
    .quality-na {color:#667085; font-weight:700;}

    [data-testid="stSidebar"] {
        background: #f4f7fb;
        border-right: 1px solid #e1e7ef;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #17365d;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        overflow: hidden;
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
    """Lunne sample-quality classification used here for OCR < 2."""
    if not np.isfinite(ocr) or ocr >= 2.0:
        return {
            "label": "Not classified",
            "css": "quality-na",
            "criterion": "Classification displayed only for OCR < 2.",
            "applicable": False,
        }

    if prediction < 0.04:
        return {
            "label": "Excellent",
            "css": "quality-excellent",
            "criterion": "Δe/e₀ < 0.04",
            "applicable": True,
        }
    if prediction < 0.07:
        return {
            "label": "Good to Fair",
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
    plt.figure(figsize=(8.0, 4.9))
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
st.caption(
    "Predict sample disturbance (Δe/e₀) using the final XGBoost model generated by the research pipeline."
)

with st.sidebar:
    st.header("Research model")
    st.success("Research outputs loaded")
    st.caption(
        "Using the saved final model and applicability-domain criteria from the research pipeline."
    )
    st.markdown("`best_model.joblib`")
    st.markdown("`applicability_domain.xlsx`")

    st.divider()
    st.subheader("Applicability criteria")
    st.write("• Observed min–max range")
    st.write("• 1st–99th percentile robust range")
    st.write("• Standardized multivariate distance")
    st.write("• Combined AD = robust range AND multivariate criterion")

    st.divider()
    st.info(
        "No model fitting, tuning, or validation is performed in this app. "
        "Predictions use the final model generated by the research pipeline."
    )


# =============================================================================
# INPUTS
# =============================================================================
st.markdown(
    '<div class="section-label">1. Input parameters</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Input fields intentionally permit values beyond the development-data range so the app can identify extrapolation."
)

cols = st.columns(5)
input_values = {}

for col, feature in zip(cols, FEATURES):
    d = domain_lookup[feature]
    default = float(d["mean"])
    span = max(d["maximum"] - d["minimum"], abs(d["maximum"]), 1.0)

    ui_min = (
        max(0.0, d["minimum"] - 0.5 * span)
        if feature != "VerticalStress"
        else 0.0
    )
    ui_max = d["maximum"] + 0.75 * span
    step = (
        0.1
        if feature in {"OCR", "AreaRatio", "CuttingEdge", "PlasticityIndex"}
        else 1.0
    )

    with col:
        input_values[feature] = st.number_input(
            FEATURE_LABELS[feature],
            min_value=float(ui_min),
            max_value=float(ui_max),
            value=float(default),
            step=float(step),
            format="%.3f" if feature == "OCR" else "%.2f",
        )
        st.caption(
            f"Observed: {d['minimum']:.3g}–{d['maximum']:.3g} | "
            f"1st–99th: {d['p01']:.3g}–{d['p99']:.3g}"
        )

predict_clicked = st.button(
    "▶  PREDICT SAMPLING DISTURBANCE",
    type="primary",
    use_container_width=True,
)


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

    st.divider()
    st.markdown(
        '<div class="section-label">2. Prediction summary</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns([1.05, 1, 1, 1.1, 1.25])
    m1.metric("Predicted Δe/e₀", f"{prediction:.4f}")
    m2.metric(
        "Observed min–max",
        "INSIDE" if ad["strict_inside"] else "OUTSIDE",
    )
    m3.metric(
        "Combined AD",
        "WITHIN" if ad["inside_combined"] else "OUTSIDE",
    )
    m4.metric(
        "Multivariate distance, D",
        f"{ad['multivariate_distance']:.3f}",
        delta=f"Threshold {ad['multivariate_threshold']:.3f}",
        delta_color="off",
    )

    with m5:
        st.markdown("**Lunne sample quality**")
        st.markdown(
            f'<div class="{quality["css"]}" style="font-size:1.45rem;">'
            f'{quality["label"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption(quality["criterion"])

    st.markdown(
        f'<div class="{ad["css"]}"><b>{ad["display_level"]}</b><br>{ad["message"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">3. Applicability-domain assessment</div>',
        unsafe_allow_html=True,
    )

    display_ad = ad["feature_table"].copy()
    numeric_cols = [
        "Input",
        "Observed min",
        "1st percentile",
        "99th percentile",
        "Observed max",
    ]
    display_ad[numeric_cols] = display_ad[numeric_cols].round(4)
    st.dataframe(
        display_ad,
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write(
            "**Robust 1st–99th percentile criterion:**",
            "PASS" if ad["robust_inside"] else "FAIL",
        )
    with c2:
        st.write(
            "**Multivariate 95th-percentile criterion:**",
            "PASS" if ad["inside_multivariate"] else "FAIL",
        )

    st.caption(
        "The combined applicability-domain flag reproduces the research pipeline: "
        "all five predictors must lie within their 1st–99th percentile ranges and "
        "the standardized multivariate distance must be below the pipeline threshold."
    )

    st.markdown(
        """
        <div class="quality-box">
            <div class="quality-title">Lunne sample-quality criteria (OCR &lt; 2)</div>
            <div><span class="quality-excellent">● Excellent:</span> Δe/e₀ &lt; 0.04</div>
            <div><span class="quality-goodfair">● Good–Fair:</span> 0.04 ≤ Δe/e₀ &lt; 0.07</div>
            <div><span class="quality-poor">● Poor:</span> 0.07 ≤ Δe/e₀ &lt; 0.14</div>
            <div><span class="quality-verypoor">● Very Poor:</span> Δe/e₀ ≥ 0.14</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">4. Local SHAP explanation</div>',
        unsafe_allow_html=True,
    )

    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    shap_sum = float(np.asarray(explanation.values[0], dtype=float).sum())
    reconstructed = base_value + shap_sum

    s1, s2, s3 = st.columns(3)
    s1.metric("SHAP baseline", f"{base_value:.4f}")
    s2.metric("Σ SHAP contributions", format_shap_value(shap_sum))
    s3.metric("Baseline + SHAP", f"{reconstructed:.4f}")

    shap_left, shap_right = st.columns([1.15, 0.85], gap="large")

    with shap_left:
        st.markdown("**SHAP waterfall plot**")
        fig = make_waterfall(explanation)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with shap_right:
        st.markdown("**SHAP contributions (local)**")
        shap_display = shap_table.drop(columns=["|SHAP|"]).copy()
        shap_display["Input"] = shap_display["Input"].map(
            lambda x: f"{float(x):.3f}"
        )
        shap_display["SHAP contribution"] = shap_display[
            "SHAP contribution"
        ].map(format_shap_value)
        st.dataframe(
            shap_display,
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Positive SHAP values increase the prediction relative to the model baseline; "
        "negative values decrease it. SHAP explains the saved model's behaviour and "
        "does not establish causality."
    )

    with st.expander("Model and research-pipeline information"):
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
        use_container_width=True,
    )


st.divider()
st.caption(
    "Research-use decision-support interface. Predictions outside the development-data "
    "applicability domain should be treated as extrapolations or weakly supported estimates "
    "and interpreted with appropriate engineering judgement."
)
