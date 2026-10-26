# Campus Facility Reservation Monitoring: Streamlit Decision-Support App

Course: Systems Analysis and Design (BS Information Systems), Section A, Finals Laboratory Exercise.
Student: Devine Aprielle G. Cabaya (Section 3A)

Turns the midterm data-mining work (predict **Reservation_Outcome**: Completed vs Not Completed) into a
Streamlit app. Predictions are decision support only; a Facility Admin records the final decision.
Uses **synthetic data only**.

## Pages
Home | Dataset (upload, schema checks, null/duplicate summary) | Data Analysis (EDA, 4 interactive Plotly charts) |
Prediction (single, batch CSV, metrics, five sample cases, midterm consistency check) |
Dashboard (observed KPIs, risk queue, recommended follow-up, admin decision log) |
System Design (5 FRs, 3 NFRs, use case diagram, workflow)

## Model (reused from the midterm)
- Decision Tree classifier inside a scikit-learn `Pipeline` (one-hot encoding + tree), saved with joblib.
- Features at the prediction checkpoint (booking submission): Facility_Type, User_Type, Capacity,
  Reservation_Duration_Hours, Lead_Time_Category.
- Excluded to avoid leakage: Usage_Status, Status, Actual_Attendance, Attendance_Ratio, Participants.
- Holdout (N=30): accuracy 66.67%, precision/recall/F1 (Completed) 77.27%; confusion matrix [[3,5],[5,17]].
- The model is loaded once with `@st.cache_resource` and never retrained on refresh.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, sign in, choose **Create app**.
3. Select the repository and branch, set the entry point to `app.py`, then **Deploy**.
4. If the build fails, open the app logs. `scikit-learn` is pinned to the version that saved the model.

## Files
```
app.py  requirements.txt  README.md  DEPLOYMENT_REPORT.md
modules/ data_processing.py exploratory_analysis.py prediction.py dashboard.py system_design.py
models/  trained_model.joblib  model_metrics.json
data/    reservations_midterm.csv  holdout_test_cases.csv  upcoming_reservations.csv
assets/  use_case_diagram.png  decision_workflow.png
tools/   build_dataset_and_model.py  make_diagrams.py
```

## Replacing the stand-in data with your real midterm files
`data/reservations_midterm.csv` is a synthetic rebuild that matches the published midterm figures
(see DEPLOYMENT_REPORT.md). To use your own notebook output, export your cleaned CSV and your fitted
pipeline with `joblib.dump(pipeline, "models/trained_model.joblib")`, then update `models/model_metrics.json`
and `data/holdout_test_cases.csv`.
