"""Exploratory Data Analysis page (interactive Plotly charts)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import TARGET

ORDER_LEAD = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
COLORS = {"Completed": "#2a9d8f", "Not Completed": "#e76f51"}


@st.cache_data
def _outcome_rate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)[TARGET].agg(
        Total="count", Uncompleted=lambda s: int((s == "Not Completed").sum())).reset_index()
    g["Uncompleted rate (%)"] = (g["Uncompleted"] / g["Total"] * 100).round(2)
    return g


def render_eda_page():
    st.header("Exploratory Data Analysis")
    df = st.session_state.get("data")
    if df is None:
        from modules.data_processing import load_default_data
        df = load_default_data()
    st.caption(f"Source: {st.session_state.get('data_source', 'Bundled synthetic midterm dataset')} "
               f"({len(df)} rows)")

    st.subheader("Descriptive statistics")
    num = df.select_dtypes("number")
    st.dataframe(num.describe().T.round(2), use_container_width=True)

    st.subheader("Class distribution")
    dist = df[TARGET].value_counts().rename_axis("Outcome").reset_index(name="Count")
    dist["Percent"] = (dist["Count"] / dist["Count"].sum() * 100).round(2)
    c1, c2 = st.columns([1, 1])
    c1.dataframe(dist, use_container_width=True, hide_index=True)
    fig = px.pie(dist, names="Outcome", values="Count", hole=0.45, color="Outcome",
                 color_discrete_map=COLORS, title="Reservation outcome")
    c2.plotly_chart(fig, use_container_width=True)
    st.caption("Imbalanced classes (about 73% Completed): accuracy alone can look better than the model is.")

    st.subheader("Interactive charts")
    tab1, tab2, tab3, tab4 = st.tabs(["Volume by facility", "Uncompleted rate", "Lead time", "Duration"])

    with tab1:
        facilities = sorted(df["Facility_Type"].unique())
        pick = st.multiselect("Facilities", facilities, default=facilities, key="eda_fac")
        sub = df[df["Facility_Type"].isin(pick)]
        if sub.empty:
            st.info("Select at least one facility.")
        else:
            vol = sub.groupby(["Facility_Type", TARGET]).size().reset_index(name="Bookings")
            st.plotly_chart(px.bar(vol, x="Bookings", y="Facility_Type", color=TARGET, orientation="h",
                                   color_discrete_map=COLORS, title="Reservation volume by facility"),
                            use_container_width=True)
            top = sub["Facility_Type"].value_counts()
            st.write(f"Highest volume: **{top.index[0]}** ({top.iloc[0]} of {len(sub)} bookings, "
                     f"{top.iloc[0] / len(sub) * 100:.2f}%).")

    with tab2:
        by = st.selectbox("Group by", ["User_Type", "Facility_Type", "Lead_Time_Category"], key="eda_by")
        g = _outcome_rate(df, by)
        g["label"] = g.apply(lambda r: f"{r['Uncompleted rate (%)']}% ({r['Uncompleted']}/{r['Total']})", axis=1)
        st.plotly_chart(px.bar(g, x=by, y="Uncompleted rate (%)", text="label", color=by,
                               title=f"Uncompleted (cancellation / no-show) rate by {by}"),
                        use_container_width=True)
        worst = g.sort_values("Uncompleted rate (%)", ascending=False).iloc[0]
        st.write(f"Highest rate: **{worst[by]}** at {worst['Uncompleted rate (%)']}% "
                 f"({worst['Uncompleted']} of {worst['Total']}).")

    with tab3:
        if "Lead_Time_Category" in df.columns:
            ct = pd.crosstab(df["Lead_Time_Category"], df[TARGET], normalize="index").mul(100).round(2)
            ct = ct.reindex([c for c in ORDER_LEAD if c in ct.index]).reset_index().melt(
                id_vars="Lead_Time_Category", var_name="Outcome", value_name="Percent")
            st.plotly_chart(px.bar(ct, x="Lead_Time_Category", y="Percent", color="Outcome",
                                   color_discrete_map=COLORS, barmode="stack",
                                   title="Outcome (%) by lead-time category"), use_container_width=True)
        else:
            st.info("No lead-time column available.")

    with tab4:
        st.plotly_chart(px.histogram(df, x="Reservation_Duration_Hours", color=TARGET, barmode="group",
                                     color_discrete_map=COLORS, nbins=10,
                                     title="Reservation duration (hours) by outcome"),
                        use_container_width=True)
        st.plotly_chart(px.box(df, x=TARGET, y="Capacity", color=TARGET, color_discrete_map=COLORS,
                               title="Requested capacity by outcome"), use_container_width=True)
