"""Data loading, schema validation and shared constants (reused from the midterm)."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "reservations_midterm.csv"
UPCOMING_PATH = ROOT / "data" / "upcoming_reservations.csv"

# Exact training features of the saved pipeline
NUMERIC_FEATURES = ["Capacity", "Reservation_Duration_Hours"]
CATEGORICAL_FEATURES = ["Facility_Type", "User_Type", "Lead_Time_Category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Reservation_Outcome"

VALID_VALUES = {
    "Facility_Type": ["Auditorium", "Conference Room", "Lab", "Lecture Hall"],
    "User_Type": ["External", "Faculty", "Staff", "Student"],
    "Lead_Time_Category": ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"],
    TARGET: ["Completed", "Not Completed"],
}
# Columns required for the EDA / dashboard (Lead_Time_Category may be derived from Lead_Time_Days)
REQUIRED_COLUMNS = ["Facility_Type", "User_Type", "Capacity", "Reservation_Duration_Hours", TARGET]

# Fields that exist only AFTER the event; they must never be used for prediction.
LEAKAGE_COLUMNS = ["Status", "Usage_Status", "Actual_Attendance", "Attendance_Ratio", "Participants",
                   TARGET]


def lead_time_category(days) -> str:
    """Same binning used in the midterm feature engineering."""
    if days <= 2:
        return "Last Minute"
    if days <= 7:
        return "Short Lead"
    if days <= 14:
        return "Medium Lead"
    return "Early Booking"


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_upcoming_data() -> pd.DataFrame:
    return pd.read_csv(UPCOMING_PATH)


def validate_dataframe(df: pd.DataFrame, need_target: bool = True):
    """Return (clean_df | None, errors, warnings). Errors block use; warnings do not."""
    errors, warnings = [], []
    if df is None or df.empty:
        return None, ["The file is empty (no rows)."], []

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = REQUIRED_COLUMNS if need_target else [c for c in REQUIRED_COLUMNS if c != TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}.")

    if "Lead_Time_Category" not in df.columns:
        if "Lead_Time_Days" in df.columns:
            days = pd.to_numeric(df["Lead_Time_Days"], errors="coerce")
            df["Lead_Time_Category"] = days.apply(lambda d: lead_time_category(d) if pd.notna(d) else None)
            warnings.append("Lead_Time_Category was derived from Lead_Time_Days.")
        else:
            errors.append("Provide Lead_Time_Category or Lead_Time_Days.")
    if errors:
        return None, errors, warnings

    for col in ["Capacity", "Reservation_Duration_Hours"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = int((df[col].isna() | (df[col] <= 0)).sum())
        if bad:
            errors.append(f"{col}: {bad} row(s) are non-numeric, empty or not positive.")

    for col, allowed in VALID_VALUES.items():
        if col == TARGET and TARGET not in df.columns:
            continue
        unknown = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        if unknown:
            errors.append(f"{col}: unexpected value(s) {unknown}. Allowed: {allowed}.")

    present_leak = [c for c in LEAKAGE_COLUMNS if c in df.columns and c != TARGET]
    if present_leak:
        warnings.append("Post-event columns found and ignored for prediction (leakage risk): "
                        + ", ".join(present_leak) + ".")
    if errors:
        return None, errors, warnings
    return df, errors, warnings


def quality_summary(df: pd.DataFrame):
    nulls = df.isna().sum().rename("Null count").to_frame()
    nulls["Null %"] = (nulls["Null count"] / len(df) * 100).round(2)
    return nulls, int(df.duplicated().sum())


def render_dataset_page():
    st.header("Dataset Upload and Validation")
    st.caption("Upload a reservation CSV, or keep using the bundled synthetic midterm dataset (150 rows).")

    with st.expander("Required schema", expanded=False):
        st.markdown(
            "**Required columns:** `Facility_Type`, `User_Type`, `Capacity`, `Reservation_Duration_Hours`, "
            "`Reservation_Outcome` (Completed / Not Completed) and either `Lead_Time_Category` or "
            "`Lead_Time_Days`.\n\nPost-event columns such as `Usage_Status`, `Actual_Attendance` and "
            "`Attendance_Ratio` are not used for prediction because they would leak the outcome.")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
            return
        except Exception as exc:  # unreadable / not a CSV
            st.error(f"Could not read the file as CSV: {exc}")
            return
        clean, errors, warnings = validate_dataframe(raw)
        for w in warnings:
            st.warning(w)
        if errors:
            for e in errors:
                st.error(e)
            st.info("The previous valid dataset stays active until a valid file is uploaded.")
            return
        st.session_state["data"] = clean
        st.session_state["data_source"] = f"Uploaded: {uploaded.name}"
        st.success(f"Validated {len(clean)} rows from {uploaded.name}.")

    df = st.session_state.get("data")
    if df is None:
        df = load_default_data()
        st.session_state["data"] = df
        st.session_state["data_source"] = "Bundled synthetic midterm dataset"
    st.write("**Active dataset:**", st.session_state["data_source"])

    nulls, dups = quality_summary(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", dups)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Null summary")
    st.dataframe(nulls, use_container_width=True)
    if dups:
        st.warning(f"{dups} duplicate row(s) detected.")
        if st.button("Remove duplicates from active dataset"):
            st.session_state["data"] = df.drop_duplicates().reset_index(drop=True)
            st.rerun()
    else:
        st.success("No duplicates, no null values." if nulls["Null count"].sum() == 0
                   else "No duplicates. Review the null summary above.")
