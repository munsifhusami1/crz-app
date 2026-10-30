# CRZ Entries Explorer (Streamlit)
# Setup:  pip install streamlit pandas requests
# Run:    streamlit run crz_app.py
# Data:   MTA Congestion Relief Zone Vehicle Entries, data.ny.gov (t6yz-b64h)

import pandas as pd
import requests
import streamlit as st

URL = "https://data.ny.gov/resource/t6yz-b64h.json"

# Aggregate on the server so we pull thousands of rows, not 6.5 million
QUERY = """
SELECT toll_date, vehicle_class,
       SUM(crz_entries) AS crz,
       SUM(excluded_roadway_entries) AS excluded
GROUP BY toll_date, vehicle_class
LIMIT 100000
"""


@st.cache_data(ttl=86400)
def load():
    r = requests.get(URL, params={"$query": QUERY}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["toll_date"] = pd.to_datetime(df["toll_date"])
    for col in ["crz", "excluded"]:
        df[col] = pd.to_numeric(df[col])
    return df.sort_values("toll_date")


st.set_page_config(page_title="CRZ Entries Explorer", layout="wide")
st.title("CRZ Entries Explorer")
st.markdown(
    "How has traffic into Manhattan's Congestion Relief Zone moved since tolling began, "
    "and which days look unusual? This app pulls the MTA's open CRZ vehicle-entries data, "
    "summed by day, and answers three questions:\n\n"
    "1. **What's the trend?** Daily entries, smoothed with a 7-day average.\n"
    "2. **Are drivers shifting to untolled roads?** The share of entries on excluded roadways.\n"
    "3. **Which days are unusual?** Days far from the normal level for that weekday.\n\n"
    "Use the sidebar to pick vehicle classes and set how strict the anomaly check is."
)
st.caption("Source: MTA Congestion Relief Zone Vehicle Entries, data.ny.gov. "
           "For traffic analysis only. The MTA notes this data is not suited to revenue estimates.")

df = load()

classes = sorted(df["vehicle_class"].unique())
picked = st.sidebar.multiselect("Vehicle class", classes, default=classes)
z_cut = st.sidebar.slider("Anomaly threshold (std devs)", 2.0, 3.5, 2.5, 0.5)

# One row per calendar day for the selected classes
d = (df[df["vehicle_class"].isin(picked)]
     .groupby("toll_date")[["crz", "excluded"]].sum()
     .asfreq("D"))  # missing dates show up as gaps, not silently skipped

# 7-day average on CALENDAR days, so a missing day can't stretch the window
d["ma_7d"] = d["crz"].rolling("7D").mean()

# Excluded-roadway share: a rough diversion signal
d["excluded_share"] = d["excluded"] / (d["crz"] + d["excluded"])

# Anomaly check: compare each day to the same weekday over the prior 6 weeks
d["dow"] = d.index.dayofweek
grp = d.groupby("dow")["crz"]
d["base_mean"] = grp.transform(lambda s: s.shift(1).rolling(6, min_periods=4).mean())
d["base_std"] = grp.transform(lambda s: s.shift(1).rolling(6, min_periods=4).std())
d["z"] = (d["crz"] - d["base_mean"]) / d["base_std"]
flags = d[d["z"].abs() > z_cut]

c1, c2, c3 = st.columns(3)
c1.metric("Days of data", f"{d['crz'].notna().sum():,}")
c2.metric("Avg daily CRZ entries", f"{d['crz'].mean():,.0f}")
c3.metric("Flagged days", len(flags))

st.subheader("1. What's the trend?")
st.caption("Daily CRZ entries, with a 7-day average to smooth out weekday swings. "
           "The average uses calendar days, so a missing day can't stretch the window.")
st.line_chart(
    d[["crz", "ma_7d"]].rename(columns={"crz": "Daily entries", "ma_7d": "7-day average"}),
    x_label="Date", y_label="Vehicle entries",
)

st.subheader("2. Are drivers shifting to untolled roads?")
st.caption("Excluded-roadway entries (e.g. the FDR Drive) as a share of all entries. "
           "A rising share could suggest diversion. It's a rough signal, not proof.")
st.line_chart(
    (d["excluded_share"] * 100).rename("Excluded-roadway share (%)"),
    x_label="Date", y_label="Share of entries (%)",
)

st.subheader("3. Which days are unusual?")
st.caption(f"Each day is compared with the same weekday over the prior six weeks. "
           f"Days more than {z_cut} standard deviations from that norm are listed. "
           f"Check for outages, holidays or weather before treating any as a real change.")
st.dataframe(
    flags[["crz", "base_mean", "z"]].round(1).rename(columns={
        "crz": "Entries that day",
        "base_mean": "Normal for that weekday",
        "z": "Std devs from normal",
    })
)

missing = d["crz"].isna().sum()
if missing:
    st.warning(f"{missing} calendar days have no data in this selection.")
