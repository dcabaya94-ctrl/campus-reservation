"""Decision-support dashboard: observed facts (Section A) vs. model forecasts (Section B)."""
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import FEATURES, TARGET, load_default_data, load_upcoming_data
from modules.prediction import (explain, load_metrics, load_model, predict_frame,
                                recommended_followup)

DECISIONS = ["Keep reservation (no action)", "Send confirmation request", "Contact user by phone/email",
             "Release slot after no confirmation", "Escalate to facility manager"]


def _filtered(df: pd.DataFrame):
    c1, c2 = st.columns(2)
    fac = c1.selectbox("Facility", ["All"] + sorted(df["Facility_Type"].unique()))
    usr = c2.selectbox("User group", ["All"] + sorted(df["User_Type"].unique()))
    if fac != "All":
        df = df[df["Facility_Type"] == fac]
    if usr != "All":
        df = df[df["User_Type"] == usr]
    return df


def render_dashboard_page():
    st.title("Campus Facility Reservation & Monitoring System")
    st.caption("Administrative Decision Dashboard")
    role = st.session_state.get("role", "Facility Admin")
    model, metrics = load_model(), load_metrics()
    hist = st.session_state.get("data")
    if hist is None:
        hist = load_default_data()

    st.subheader("Filters")
    st.caption("Filters apply to Section A. The dataset has no booking dates, so a date-range filter is not available.")
    view = _filtered(hist)
    if view.empty:
        st.warning("No reservations match the selected filters.")
        return

    # ------------------ Section A: observed ------------------
    st.subheader("A. Historical operational metrics (observed facts)")
    total = len(view)
    done = int((view[TARGET] == "Completed").sum())
    bad = total - done
    top_fac = view["Facility_Type"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total bookings", total, f"Most loaded: {top_fac.index[0]} ({top_fac.iloc[0]})", delta_color="off")
    k2.metric("Completed", f"{done} ({done / total * 100:.2f}%)")
    k3.metric("Not completed", f"{bad} ({bad / total * 100:.2f}%)")
    rates = (view.groupby("User_Type")[TARGET].apply(lambda s: (s == "Not Completed").mean() * 100)
             .sort_values(ascending=False))
    k4.metric("Highest uncompleted rate", f"{rates.index[0]} {rates.iloc[0]:.2f}%",
              f"denominator: {int((view['User_Type'] == rates.index[0]).sum())} bookings", delta_color="off")
    st.caption(f"Observed facts. Percentages use {total} bookings as the denominator.")
    chart = rates.round(2).rename_axis("User group").reset_index(name="Uncompleted rate (%)")
    st.plotly_chart(px.bar(chart, x="User group", y="Uncompleted rate (%)", text="Uncompleted rate (%)",
                           title="Cancellation / no-show % by user group (observed)"),
                    use_container_width=True)

    # ------------------ Section B: predicted ------------------
    st.subheader("B. Predictive risk monitoring (statistical forecast, next 7 days)")
    st.warning("These are model estimates, not observed facts. They never approve or reject a reservation.")
    batch = st.session_state.get("scored_batch")
    if batch is not None and "P_Not_Completed" in batch.columns:
        queue = batch.copy()
        st.caption("Source: batch scored on the Prediction page.")
    else:
        up = load_upcoming_data()
        queue = predict_frame(model, up)
        keep = ["Reservation_ID"] + FEATURES + ["P_Not_Completed", "Predicted_Outcome", "Risk_Tier"]
        queue = queue[keep]
        st.caption("Source: bundled synthetic upcoming reservations. Score your own on the Prediction page.")
    if "Reservation_ID" not in queue.columns:
        queue.insert(0, "Reservation_ID", [f"ROW{i + 1}" for i in range(len(queue))])

    tiers = queue["Risk_Tier"].value_counts()
    t1, t2, t3 = st.columns(3)
    t1.metric("HIGH risk", int(tiers.get("HIGH", 0)))
    t2.metric("MEDIUM risk", int(tiers.get("MEDIUM", 0)))
    t3.metric("LOW risk", int(tiers.get("LOW", 0)))

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    queue = queue.sort_values(["Risk_Tier", "P_Not_Completed"],
                              key=lambda s: s.map(order) if s.name == "Risk_Tier" else -s).reset_index(drop=True)
    show = queue.copy()
    show["Chance of no-show"] = (show["P_Not_Completed"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(show.drop(columns=["P_Not_Completed"]), use_container_width=True, hide_index=True)

    st.subheader("Interpretation and recommended human follow-up")
    flagged = queue[queue["Risk_Tier"].isin(["HIGH", "MEDIUM"])]
    if flagged.empty:
        st.success("No reservations flagged for review.")
    for _, r in flagged.head(8).iterrows():
        shown, top, br = explain(model, r, hist)
        with st.expander(f"{r['Reservation_ID']} | {r['Facility_Type']} | {r['User_Type']} | "
                         f"{r['Risk_Tier']} ({r['P_Not_Completed'] * 100:.0f}% estimated no-show)"):
            st.write("**Contributing predictors:** " + "; ".join(shown))
            st.write("**Historical base rates:** " + "; ".join(f"{k}: {v}" for k, v in br.items()))
            st.write("**Follow-up:** " + recommended_followup(r["Risk_Tier"], r["User_Type"]))

    # ------------------ human decision ------------------
    st.subheader("Final human decision")
    if role != "Facility Admin":
        st.info("Only a Facility Admin can record final decisions. Switch the role in the sidebar "
                "(demo only; production would use institutional login).")
    elif flagged.empty:
        st.caption("Nothing to decide right now.")
    else:
        with st.form("decision"):
            rid = st.selectbox("Reservation", flagged["Reservation_ID"])
            who = st.text_input("Reviewer name")
            dec = st.selectbox("Decision", DECISIONS)
            note = st.text_area("Reason / note (required)")
            ok = st.form_submit_button("Record decision")
        if ok:
            if not who.strip() or not note.strip():
                st.error("Reviewer name and a note are required.")
            else:
                row = flagged[flagged["Reservation_ID"] == rid].iloc[0]
                log = st.session_state.setdefault("decision_log", [])
                log.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Reservation_ID": rid,
                            "Model_Risk": row["Risk_Tier"], "Model_Version": metrics["model_version"],
                            "Decision": dec, "Reviewer": who.strip(), "Note": note.strip()})
                st.success(f"Decision recorded for {rid}.")
    log = st.session_state.get("decision_log", [])
    if log:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
        st.download_button("Download decision log", pd.DataFrame(log).to_csv(index=False),
                           "decision_log.csv", "text/csv")

    st.caption(f"Model: {metrics['model_version']} | Accuracy {metrics['accuracy'] * 100:.2f}% | "
               "Advisory output; a human makes the final decision.")
