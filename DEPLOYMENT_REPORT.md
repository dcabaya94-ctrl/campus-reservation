# Deployment Report

## Reused code
Problem, target (Reservation_Outcome), predictors, Lead_Time_Category bins (0-2 Last Minute, 3-7 Short Lead,
8-14 Medium Lead, >14 Early Booking), 80/20 stratified split with fixed random_state, Decision Tree
classifier, metrics, EDA questions, FR-DM-01..03, use case diagram and dashboard wireframe from the midterm.

## Changes from the midterm
- Training moved out of the app: the fitted preprocessing + tree pipeline is saved with joblib.
- Midterm requirements FR-DM-01 and FR-DM-02 (automatic booking restriction, automatic slot release) are
  narrowed to monitoring and recommendation, so nothing is rejected or released without human review.
- Added FR-DM-04 (CSV validation) and FR-DM-05 (admin decision log); use case diagram updated to match.
- The wireframe's Date Range filter is not implemented because the dataset has no booking dates.

## Data and model provenance (read this)
The original midterm CSV and notebook were not available to the build, so `data/reservations_midterm.csv`
is a **synthetic rebuild** that matches: 150 rows and 14 columns; facility counts (46/36/34/34); user counts
(39/46/30/35); uncompleted by user (10/10/13/7 = 40); Completed 110 / Not Completed 40; the Early Booking,
Last Minute and Medium Lead outcome percentages; and the holdout confusion matrix, accuracy, precision, recall
and F1. Known differences: the midterm lead-time table sums to 143 rows, so Short Lead is 81.25% here versus
84.00% in the midterm; the feature-importance ranking differs from the midterm table; and the tree depth
(max_depth=5) is assumed. **Swap in your real CSV and pipeline before submitting** (see README).

## Tests
| Test | Result |
|---|---|
| Empty CSV, missing columns, invalid categories, negative capacity | Clear error messages (verified in code) |
| Lead_Time_Category derived from Lead_Time_Days | Works |
| Saved pipeline vs midterm holdout | 30/30 identical predictions, accuracy 66.67% |
| Single prediction input types | Match training feature names and dtypes |
| Risk tiers on 18 upcoming bookings | 5 HIGH, 1 MEDIUM, 12 LOW |
| Streamlit pages in a browser, deployed URL, screenshots | **Not yet run** |

## Limitations
150 rows (30 test), so metrics are unstable; accuracy 66.67% means advisory use only; a single tree gives
coarse probabilities (many 0% or 100%); no date, reminder or exam-week features; role selector is a demo,
not real authentication; confirmation prompts are recommendations, not sent messages.

## Reflection
(Write your own: what was hardest about moving from notebook to app, and what you would improve.)
