"""Model loading and inference. Uses the pipeline saved from the midterm (no retraining)."""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import (FEATURES, TARGET, VALID_VALUES, lead_time_category,
                                     load_default_data, validate_dataframe)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
HOLDOUT_PATH = ROOT / "data" / "holdout_test_cases.csv"

HIGH_RISK, MEDIUM_RISK = 0.50, 0.35  # thresholds on P(Not Completed)

LIMITATIONS = [
    "Trained on only 150 synthetic reservations (120 train / 30 test): metrics move a lot with small data shifts.",
    "Accuracy is 66.67%, so the output is an advisory risk flag and must not run on autopilot.",
    "Probabilities come from a single decision tree (leaf class proportions) and are coarse, not calibrated guarantees.",
    "Only booking-time features are used (facility, user type, capacity, duration, lead time). "
    "Reminders, weather, exams and day of week are not captured.",
    "Predictions never approve or reject a reservation; a Facility Admin records the final decision.",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def risk_tier(p_not_completed: float) -> str:
    if p_not_completed >= HIGH_RISK:
        return "HIGH"
    if p_not_completed >= MEDIUM_RISK:
        return "MEDIUM"
    return "LOW"


def build_input_df(facility, user, capacity, duration, lead_days) -> pd.DataFrame:
    """One-row frame with the exact training feature names and types."""
    return pd.DataFrame([{
        "Capacity": int(capacity),
        "Reservation_Duration_Hours": int(duration),
        "Facility_Type": str(facility),
        "User_Type": str(user),
        "Lead_Time_Category": lead_time_category(int(lead_days)),
    }])[FEATURES]


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Return frame + probabilities, predicted label and risk tier."""
    out = frame.copy()
    proba = model.predict_proba(frame[FEATURES])
    classes = list(model.classes_)
    out["P_Completed"] = proba[:, classes.index("Completed")].round(4)
    out["P_Not_Completed"] = proba[:, classes.index("Not Completed")].round(4)
    out["Predicted_Outcome"] = model.predict(frame[FEATURES])
    out["Risk_Tier"] = out["P_Not_Completed"].apply(risk_tier)
    return out


def base_rates(df: pd.DataFrame, row: pd.Series) -> dict:
    rates = {}
    for col in ["User_Type", "Facility_Type", "Lead_Time_Category"]:
        if col in df.columns:
            sub = df[df[col] == row[col]]
            if len(sub):
                k = int((sub[TARGET] == "Not Completed").sum())
                rates[f"{col} = {row[col]}"] = f"{k / len(sub) * 100:.1f}% uncompleted ({k}/{len(sub)})"
    return rates


def explain(model, row: pd.Series, hist: pd.DataFrame):
    """Contributing predictors, global importance and historical base rates."""
    metrics = load_metrics()
    imp = metrics["feature_importances"]
    shown = [f"{c}: {row[c]}" for c in FEATURES]
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
    return shown, top, base_rates(hist, row)


def recommended_followup(tier: str, user: str) -> str:
    if tier == "HIGH":
        extra = " (Staff: send 48-hour confirmation request, release the slot if no reply in 12 hours)" \
            if user == "Staff" else ""
        return "Queue for Facility Admin review; send a confirmation request" + extra + "."
    if tier == "MEDIUM":
        return "Send an automated reminder 48 hours before the slot; no manual action required."
    return "No action. Keep the standard booking workflow."


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_prediction_page():
    st.header("Predictive Modeling")
    model = load_model()
    metrics = load_metrics()
    hist = st.session_state.get("data")
    if hist is None:
        hist = load_default_data()

    st.info("Decision support only. A high-risk flag never rejects a reservation automatically; "
            "a Facility Admin reviews it and records the final decision.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model version", metrics["model_version"])
    m2.metric("Algorithm", "Decision Tree")
    m3.metric("Trained", metrics["trained_at"])

    tab1, tab2, tab3, tab4 = st.tabs(["Single prediction", "Batch (CSV)", "Model metrics",
                                      "Sample cases & consistency"])

    # ---------------- single -----------------
    with tab1:
        with st.form("single"):
            c1, c2 = st.columns(2)
            facility = c1.selectbox("Facility type", VALID_VALUES["Facility_Type"])
            user = c2.selectbox("User role", VALID_VALUES["User_Type"])
            capacity = c1.number_input("Capacity requested", min_value=1, max_value=1000, value=50, step=10)
            duration = c2.number_input("Duration (hours)", min_value=1, max_value=24, value=2)
            lead = c1.number_input("Lead time (days between booking and use)", min_value=0, max_value=365,
                                   value=10)
            go = st.form_submit_button("Predict")
        if go:
            if duration > 12 or lead > 120 or capacity > 500:
                st.error("Inputs are far outside realistic campus bookings. Please check the values.")
            else:
                if duration > 5 or lead > 30 or capacity > 200:
                    st.warning("Some inputs are outside the range seen in training (duration up to 5 h, "
                               "lead time up to 30 days, capacity up to 200). Treat the result with extra caution.")
                x = build_input_df(facility, user, capacity, duration, lead)
                res = predict_frame(model, x).iloc[0]
                st.session_state["last_prediction"] = (res.to_dict(), _stamp())
        if "last_prediction" in st.session_state:
            res, ts = st.session_state["last_prediction"]
            res = pd.Series(res)
            a, b, c = st.columns(3)
            a.metric("Predicted outcome", res["Predicted_Outcome"])
            b.metric("P(Completed)", f"{res['P_Completed'] * 100:.1f}%")
            c.metric("P(Not Completed)", f"{res['P_Not_Completed'] * 100:.1f}%")
            color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[res["Risk_Tier"]]
            st.markdown(f"**Risk tier:** :{color}[{res['Risk_Tier']}]  \n"
                        f"**Recommended human follow-up:** "
                        f"{recommended_followup(res['Risk_Tier'], res['User_Type'])}")
            shown, top, rates = explain(model, res, hist)
            with st.expander("Why this prediction? (contributing predictors)", expanded=True):
                st.write("**Inputs used:** " + "; ".join(shown))
                st.write("**Most influential predictors in the model:** "
                         + ", ".join(f"{k} ({v:.3f})" for k, v in top))
                st.write("**Historical base rates in the active dataset:**")
                for k, v in rates.items():
                    st.write(f"- {k}: {v}")
            st.caption(f"Model: {metrics['model_version']} | Prediction time: {ts} | "
                       f"Pending Facility Admin review.")

    # ---------------- batch -----------------
    with tab2:
        st.write("Upload future reservations with the five booking-time fields "
                 "(`Facility_Type`, `User_Type`, `Capacity`, `Reservation_Duration_Hours`, and "
                 "`Lead_Time_Days` or `Lead_Time_Category`). No outcome column is needed.")
        up = st.file_uploader("Reservations to score", type="csv", key="batch_up")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
                raw = None
            if raw is not None:
                clean, errors, warnings = validate_dataframe(raw, need_target=False)
                for w in warnings:
                    st.warning(w)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    scored = predict_frame(model, clean)
                    keep = (["Reservation_ID"] if "Reservation_ID" in scored else []) + FEATURES + \
                        ["P_Not_Completed", "Predicted_Outcome", "Risk_Tier"]
                    st.session_state["scored_batch"] = scored[keep]
                    st.session_state["scored_batch_time"] = _stamp()
        if "scored_batch" in st.session_state:
            st.dataframe(st.session_state["scored_batch"], use_container_width=True)
            st.caption(f"Scored {st.session_state['scored_batch_time']} with {metrics['model_version']}. "
                       "Scored batches appear in the Dashboard risk queue.")
            st.download_button("Download predictions", st.session_state["scored_batch"].to_csv(index=False),
                               "predictions.csv", "text/csv")

    # ---------------- metrics -----------------
    with tab3:
        st.write(f"**Algorithm:** {metrics['algorithm']}  \n**Split:** {metrics['split']}  \n"
                 f"**Positive class:** {metrics['positive_class']}  \n"
                 f"**Features:** {', '.join(metrics['features'])}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        k2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        k3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        k4.metric("F1-score", f"{metrics['f1'] * 100:.2f}%")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"Actual {l}" for l in metrics["confusion_labels"]],
                          columns=[f"Pred {l}" for l in metrics["confusion_labels"]])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  title="Confusion matrix (holdout, N = 30)"), use_container_width=True)
        imp = pd.Series(metrics["feature_importances"]).sort_values().tail(10).rename_axis("Feature") \
            .reset_index(name="Importance")
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="Feature importances (saved pipeline)"), use_container_width=True)
        st.subheader("Limitations")
        for item in LIMITATIONS:
            st.write(f"- {item}")

    # ---------------- sample & consistency -----------------
    with tab4:
        st.subheader("Five sample predictions")
        samples = pd.DataFrame([
            ("Auditorium", "Staff", 200, 4, 22), ("Lab", "Staff", 50, 3, 15),
            ("Lecture Hall", "External", 100, 3, 19), ("Conference Room", "Student", 20, 1, 5),
            ("Lab", "Faculty", 50, 2, 2)],
            columns=["Facility_Type", "User_Type", "Capacity", "Reservation_Duration_Hours", "Lead_Time_Days"])
        samples["Lead_Time_Category"] = samples["Lead_Time_Days"].apply(lead_time_category)
        out = predict_frame(model, samples[FEATURES].assign(Lead_Time_Days=samples["Lead_Time_Days"]))
        st.dataframe(out.drop(columns=["P_Completed"]), use_container_width=True)

        st.subheader("Consistency with the midterm notebook")
        st.write("The holdout cases below are the same 30 rows from the midterm split. The saved pipeline must "
                 "reproduce the midterm predictions and confusion matrix.")
        if st.button("Run consistency check"):
            hold = pd.read_csv(HOLDOUT_PATH)
            pred = model.predict(hold[FEATURES])
            match = int((pred == hold["Midterm_Prediction"]).sum())
            cmx = pd.crosstab(hold["Actual_Outcome"], pred).reindex(
                index=["Not Completed", "Completed"], columns=["Not Completed", "Completed"], fill_value=0)
            acc = (pred == hold["Actual_Outcome"]).mean()
            if match == len(hold):
                st.success(f"{match}/{len(hold)} predictions identical to the midterm. "
                           f"Accuracy {acc * 100:.2f}%.")
            else:
                st.error(f"Only {match}/{len(hold)} predictions match the midterm.")
            st.dataframe(cmx)
