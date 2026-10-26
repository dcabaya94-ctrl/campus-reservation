import streamlit as st

from modules.dashboard import render_dashboard_page
from modules.data_processing import load_default_data, render_dataset_page
from modules.exploratory_analysis import render_eda_page
from modules.prediction import load_metrics, render_prediction_page
from modules.system_design import render_system_design_page

st.set_page_config(page_title="Campus Reservation Decision Support", layout="wide")

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
menu = st.sidebar.selectbox("Module", PAGES)
st.sidebar.selectbox("Acting as (demo role)", ["Facility Admin", "Reserver"], key="role",
                     help="Only a Facility Admin may record final decisions.")
st.sidebar.caption("Synthetic data only. Predictions support, never replace, human decisions.")

if "data" not in st.session_state:
    st.session_state["data"] = load_default_data()
    st.session_state["data_source"] = "Bundled synthetic midterm dataset"

if menu == "Home":
    metrics = load_metrics()
    st.title("Campus Facility Reservation Monitoring")
    st.subheader("Predicting cancellation and no-show to support facility administrators")
    st.write(
        "This app turns the midterm data-mining work into a working information-system module: "
        "it reuses the Decision Tree pipeline trained for **Reservation_Outcome** "
        "(Completed vs Not Completed) and shows how its predictions feed an authorized human review.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservations", len(st.session_state["data"]))
    c2.metric("Model accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c3.metric("Precision (Completed)", f"{metrics['precision'] * 100:.2f}%")
    c4.metric("F1-score", f"{metrics['f1'] * 100:.2f}%")
    st.markdown(
        "**Modules:** Dataset (upload and validate) | Data Analysis (EDA) | Prediction | "
        "Dashboard (risk queue and decisions) | System Design (requirements and diagrams)")
    st.info("Prediction checkpoint: the moment a reservation is submitted. Only booking-time features are used; "
            "post-event fields (Usage_Status, Actual_Attendance, Attendance_Ratio) are excluded to prevent leakage.")
    st.warning("Predictions are decision-support outputs, not automatic authorization to reject a reservation "
               "or deny student services.")
elif menu == "Dataset":
    render_dataset_page()
elif menu == "Data Analysis":
    render_eda_page()
elif menu == "Prediction":
    render_prediction_page()
elif menu == "Dashboard":
    render_dashboard_page()
else:
    render_system_design_page()
