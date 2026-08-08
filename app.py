from __future__ import annotations

from pathlib import Path
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

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
    "AreaRatio": "Area ratio, AR (%)",
    "CuttingEdge": "Cutting-edge angle, CE (°)",
    "PlasticityIndex": "Plasticity index, PI (%)",
    "OCR": "Overconsolidation ratio, OCR",
    "VerticalStress": "In-situ vertical effective stress, σ′v₀ (kPa)",
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

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px;}
    .small-note {color:#5f6368; font-size:0.90rem;}
    .status-good {padding:0.85rem 1rem; border-left:5px solid #2e7d32; background:#eef8ef; border-radius:8px;}
    .status-warn {padding:0.85rem 1rem; border-left:5px solid #ed9b00; background:#fff8e8; border-radius:8px;}
    .status-bad {padding:0.85rem 1rem; border-left:5px solid #c62828; background:#fff0f0; border-radius:8px;}
    div[data-testid="stMetric"] {border:1px solid #e1e4e8; padding:0.8rem; border-radius:12px; background:#ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)


def discover_pipeline_output_dir() -> Path | None:
    """Find a directory containing the exact outputs from the research pipeline."""
    candidates: list[Path] = []

    # Preferred explicit folder.
    explicit = Path("pipeline_outputs")
    if explicit.is_dir():
        candidates.append(explicit)

    # Current working directory.
    candidates.append(Path("."))

    # Timestamped Results_* folders produced by the research pipeline.
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
        "Feature", "Minimum", "Percentile_1", "Percentile_99",
        "Maximum", "Mean", "Standard_Deviation"
    }
    missing = required_range_cols.difference(ranges.columns)
    if missing:
        raise ValueError(
            "applicability_domain.xlsx is missing expected Predictor_Ranges columns: "
            + ", ".join(sorted(missing))
        )

    missing_features = [f for f in FEATURES if f not in set(ranges["Feature"].astype(str))]
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
            raise ValueError(f"Invalid standard deviation for {feature} in applicability_domain.xlsx")

        z = (value - d["mean"]) / d["std"]
        z_values.append(z)

        if not inside_minmax:
            status = "Outside observed min–max"
        elif not inside_robust:
            status = "Within min–max; outside 1st–99th percentile"
        else:
            status = "Within 1st–99th percentile"

        rows.append({
            "Parameter": FEATURE_LABELS[feature],
            "Input": value,
            "Observed min": d["minimum"],
            "1st percentile": d["p01"],
            "99th percentile": d["p99"],
            "Observed max": d["maximum"],
            "Status": status,
        })

    multivariate_distance = float(np.sqrt(np.square(z_values).sum()))
    inside_multivariate = multivariate_distance <= float(multivariate_threshold)

    # This reproduces the research pipeline's combined applicability-domain rule:
    # robust 1st–99th percentile domain AND multivariate 95th-percentile domain.
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
            reasons.append("one or more predictors are outside the 1st–99th percentile range")
        if not inside_multivariate:
            reasons.append("the standardized multivariate distance exceeds the pipeline threshold")
        message = "Prediction should be interpreted cautiously because " + " and ".join(reasons) + "."
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


def make_shap_table(input_df, explanation):
    values = np.asarray(explanation.values[0], dtype=float)
    rows = []
    for i, feature in enumerate(FEATURES):
        shap_value = float(values[i])
        rows.append({
            "Parameter": FEATURE_LABELS[feature],
            "Input": float(input_df.iloc[0][feature]),
            "SHAP contribution": shap_value,
            "Direction": (
                "Increases prediction" if shap_value > 0
                else "Decreases prediction" if shap_value < 0
                else "Neutral"
            ),
            "|SHAP|": abs(shap_value),
        })
    return pd.DataFrame(rows).sort_values("|SHAP|", ascending=False).reset_index(drop=True)


def make_waterfall(explanation):
    plt.figure(figsize=(8.8, 5.8))
    shap.plots.waterfall(explanation[0], max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def make_excel_report(input_df, prediction, ad, shap_table, pipeline_dir):
    base_value = np.nan
    summary = pd.DataFrame({
        "Metric": [
            "Predicted SampleDisturbance (Δe/e0)",
            "Observed min-max domain",
            "Robust 1st-99th percentile domain",
            "Standardized multivariate distance",
            "Multivariate distance threshold",
            "Inside multivariate 95th-percentile domain",
            "Inside combined applicability domain",
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
            str(pipeline_dir),
        ],
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="Inputs", index=False)
        summary.to_excel(writer, sheet_name="Prediction", index=False)
        ad["feature_table"].to_excel(writer, sheet_name="Applicability", index=False)
        shap_table.drop(columns=["|SHAP|"]).to_excel(writer, sheet_name="Local_SHAP", index=False)
    return output.getvalue()


# -----------------------------------------------------------------------------
# Load the exact outputs produced by the user's research pipeline
# -----------------------------------------------------------------------------
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

# Check that the saved model is compatible with the expected research predictors.
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

st.title("Soil Sampling Disturbance Prediction")
st.caption(
    "Deployment interface for the final XGBoost model generated by the research pipeline. "
    "No model fitting, tuning, or validation is performed in this app."
)

with st.sidebar:
    st.header("Pipeline source")
    st.success("Research outputs loaded")
    st.caption(str(pipeline_dir))
    st.markdown(
        "The GUI uses the saved `best_model.joblib` and the applicability-domain "
        "criteria exported to `applicability_domain.xlsx`."
    )

    st.divider()
    st.subheader("Applicability criteria")
    st.write("Observed min–max range")
    st.write("1st–99th percentile robust range")
    st.write("Standardized multivariate distance")
    st.write("Combined AD = robust range AND multivariate criterion")

st.subheader("Input parameters")
st.caption(
    "Input fields intentionally permit values beyond the development-data range so the app can identify extrapolation."
)

cols = st.columns(5)
input_values = {}

for col, feature in zip(cols, FEATURES):
    d = domain_lookup[feature]
    default = float(d["mean"])
    span = max(d["maximum"] - d["minimum"], abs(d["maximum"]), 1.0)
    # Wider UI bounds are only interface limits; they are not model applicability limits.
    ui_min = max(0.0, d["minimum"] - 0.5 * span) if feature != "VerticalStress" else 0.0
    ui_max = d["maximum"] + 0.75 * span
    step = 0.1 if feature in {"OCR", "AreaRatio", "CuttingEdge", "PlasticityIndex"} else 1.0

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

predict_clicked = st.button("Predict sampling disturbance", type="primary", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)

    prediction = float(model.predict(input_df)[0])
    ad = evaluate_applicability(input_df, domain_lookup, multivariate_threshold)
    explanation = explainer(input_df)
    shap_table = make_shap_table(input_df, explanation)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Δe/e₀", f"{prediction:.4f}")
    m2.metric("Observed min–max", "Inside" if ad["strict_inside"] else "Outside")
    m3.metric("Combined AD", "Inside" if ad["inside_combined"] else "Outside")
    m4.metric(
        "Multivariate distance",
        f"{ad['multivariate_distance']:.3f}",
        delta=f"Threshold {ad['multivariate_threshold']:.3f}",
        delta_color="off",
    )

    st.markdown(
        f'<div class="{ad["css"]}"><b>{ad["display_level"]}</b><br>{ad["message"]}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Applicability-domain assessment")
    display_ad = ad["feature_table"].copy()
    numeric_cols = ["Input", "Observed min", "1st percentile", "99th percentile", "Observed max"]
    display_ad[numeric_cols] = display_ad[numeric_cols].round(4)
    st.dataframe(display_ad, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Robust 1st–99th percentile criterion:**", "PASS" if ad["robust_inside"] else "FAIL")
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

    st.subheader("Local SHAP explanation")

    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    shap_sum = float(np.asarray(explanation.values[0], dtype=float).sum())
    reconstructed = base_value + shap_sum

    s1, s2, s3 = st.columns(3)
    s1.metric("SHAP baseline", f"{base_value:.4f}")
    s2.metric("Σ SHAP contributions", f"{shap_sum:+.4f}")
    s3.metric("Baseline + SHAP", f"{reconstructed:.4f}")

    fig = make_waterfall(explanation)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    shap_display = shap_table.drop(columns=["|SHAP|"]).copy()
    shap_display["Input"] = shap_display["Input"].round(4)
    shap_display["SHAP contribution"] = shap_display["SHAP contribution"].round(6)
    st.dataframe(shap_display, use_container_width=True, hide_index=True)

    st.info(
        "Positive SHAP values increase the prediction relative to the model baseline; "
        "negative values decrease it. SHAP explains the saved model's behaviour and "
        "does not establish causality."
    )

    with st.expander("Model and research-pipeline information"):
        st.write("**Loaded model:**", str(pipeline_dir / "best_model.joblib"))
        st.write("**Loaded applicability domain:**", str(pipeline_dir / "applicability_domain.xlsx"))

        best_summary = optional_outputs.get("best_model_summary.xlsx")
        if isinstance(best_summary, pd.DataFrame) and not best_summary.empty:
            st.write("**Best model summary**")
            st.dataframe(best_summary, use_container_width=True, hide_index=True)

        settings = optional_outputs.get("reproducibility_settings.xlsx")
        if isinstance(settings, pd.DataFrame) and not settings.empty:
            st.write("**Reproducibility settings**")
            st.dataframe(settings, use_container_width=True, hide_index=True)

        versions = optional_outputs.get("software_versions.json")
        if isinstance(versions, dict):
            st.write("**Software versions used by research pipeline**")
            st.json(versions)

        params = optional_outputs.get("best_model_parameters.json")
        if isinstance(params, dict):
            st.write("**Saved final hyperparameters**")
            st.json(params)

    report_bytes = make_excel_report(input_df, prediction, ad, shap_table, pipeline_dir)
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
