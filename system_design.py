"""System Design page: requirements, actors, use case diagram and decision workflow."""
from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets"

FUNCTIONAL = pd.DataFrame([
    ("FR-DM-01", "Part III EDA: Auditorium has the highest load (46 of 150 bookings, 30.67%).",
     "The system shall display reservation volume per facility and highlight the most heavily loaded facility "
     "so administrators can review load balancing.", "Data Analysis, Dashboard"),
    ("FR-DM-02", "Part III EDA: Staff have the highest uncompleted rate (43.33%, 13 of 30).",
     "The system shall recommend a confirmation prompt to the admin for high-risk Staff reservations "
     "(48 h before the slot; release after 12 h without reply). The prompt is shown as a recommended "
     "follow-up; sending real email/SMS is out of scope.", "Prediction, Dashboard"),
    ("FR-DM-03", "Part IV model: Lead_Time_Category (Early Booking) is a leading predictor; accuracy 66.67%.",
     "The system shall compute a risk tier (HIGH / MEDIUM / LOW) from the saved model for each reservation "
     "and place HIGH and MEDIUM ones in the admin review queue.", "Prediction, Dashboard"),
    ("FR-DM-04", "Part II data-quality summary and Task 3 validation rule: bad data corrupts analytics.",
     "The system shall validate uploaded CSV files (schema, empty file, invalid values, nulls, duplicates) "
     "and show clear error messages without breaking the app.", "Dataset"),
    ("FR-DM-05", "Task 12(a): predicted high risk must not auto-reject a reservation.",
     "The system shall allow only a Facility Admin to record the final decision, with reviewer, reason, "
     "time and model version in a downloadable log.", "Dashboard"),
], columns=["ID", "Finding / evidence", "Requirement (the system shall ...)", "Implemented in"])

NONFUNCTIONAL = pd.DataFrame([
    ("NFR-01 Performance", "The model is loaded once (cached) and never retrained on page refresh; "
     "single predictions return within about one second."),
    ("NFR-02 Transparency", "Every prediction shows model version, timestamp, probability, contributing "
     "predictors, historical base rates and limitations."),
    ("NFR-03 Privacy and security", "Only synthetic or authorized data is used; no credentials or real "
     "student records are committed; decision recording is restricted to the Facility Admin role."),
], columns=["ID", "Requirement"])


def render_system_design_page():
    st.header("System Design (SAD Integration)")
    st.subheader("Functional requirements")
    st.dataframe(FUNCTIONAL, use_container_width=True, hide_index=True)
    st.caption("FR-DM-01 and FR-DM-02 are narrowed from the midterm wording (automatic booking restriction and "
               "automatic release) to monitoring and recommendation, so that nothing is rejected or released "
               "without human review.")
    st.subheader("Non-functional requirements")
    st.dataframe(NONFUNCTIONAL, use_container_width=True, hide_index=True)

    st.subheader("Actors")
    st.markdown(
        "- **Reserver** (Student, Faculty, Staff, External): submits booking details and sees the prediction "
        "for their own booking. Cannot record decisions.\n"
        "- **Predictive Engine** (system actor): the saved Decision Tree pipeline that scores each booking.\n"
        "- **Facility Admin**: views the dashboard and risk queue, and is the **only** actor who records the "
        "final decision.")

    st.subheader("Updated use case diagram")
    st.image(str(ASSETS / "use_case_diagram.png"), use_container_width=True)
    st.caption("UC-01 includes UC-02 (every booking is risk-assessed). UC-03 extends UC-02 only for "
               "high-risk Staff bookings. UC-04 includes UC-05 (a reviewer must see the explanation).")

    st.subheader("Decision-support workflow")
    st.image(str(ASSETS / "decision_workflow.png"), use_container_width=True)

    st.subheader("Analytical process flow")
    st.markdown("Historical data -> validation and cleaning -> exploratory analysis -> saved model "
                "(trained once, midterm) -> prediction at booking time -> risk tier -> human review -> "
                "recorded decision.")
