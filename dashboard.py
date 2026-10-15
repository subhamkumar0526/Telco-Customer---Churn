"""
Interactive sales dashboard.   Run:  streamlit run dashboard.py
(Run generate_data.py and analysis.py first so data/sales_clean.csv exists.)
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sales Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load():
    return pd.read_csv("data/sales_clean.csv", parse_dates=["order_date", "month_start"])


df = load()
st.title("📊 Retail Sales Dashboard")

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
years = st.sidebar.multiselect("Year", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
cats = st.sidebar.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
segments = st.sidebar.multiselect("Segment", sorted(df["segment"].unique()), default=sorted(df["segment"].unique()))

f = df[
    df["year"].isin(years)
    & df["region"].isin(regions)
    & df["category"].isin(cats)
    & df["segment"].isin(segments)
]
if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------- KPI row ----------------
sales, profit = f["sales"].sum(), f["profit"].sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sales", f"{sales:,.0f}")
c2.metric("Profit", f"{profit:,.0f}")
c3.metric("Margin", f"{profit / sales * 100:.1f}%")
c4.metric("Orders", f"{f['order_id'].nunique():,}")
c5.metric("Customers", f"{f['customer_id'].nunique():,}")

# ---------------- charts ----------------
monthly = f.groupby("month_start", as_index=False)[["sales", "profit"]].sum()
st.plotly_chart(
    px.line(monthly, x="month_start", y=["sales", "profit"], markers=True, title="Monthly Sales & Profit"),
    use_container_width=True,
)

left, right = st.columns(2)
cat = f.groupby("category", as_index=False)[["sales", "profit"]].sum()
left.plotly_chart(
    px.bar(cat, x="category", y=["sales", "profit"], barmode="group", title="By Category"),
    use_container_width=True,
)
reg = f.groupby("region", as_index=False)[["sales", "profit"]].sum()
right.plotly_chart(
    px.bar(reg, x="region", y=["sales", "profit"], barmode="group", title="By Region"),
    use_container_width=True,
)

left, right = st.columns(2)
sub = f.groupby("sub_category", as_index=False)["profit"].sum().sort_values("profit")
left.plotly_chart(
    px.bar(sub, x="profit", y="sub_category", orientation="h", title="Profit by Sub-category",
           color="profit", color_continuous_scale="RdYlGn"),
    use_container_width=True,
)
disc = f.groupby("discount", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
disc["margin_pct"] = disc["profit"] / disc["sales"] * 100
right.plotly_chart(
    px.bar(disc, x="discount", y="margin_pct", title="Margin % by Discount",
           color="margin_pct", color_continuous_scale="RdYlGn"),
    use_container_width=True,
)

st.subheader("Top 10 customers")
top = (
    f.groupby("customer_id")
    .agg(sales=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "count"))
    .sort_values("sales", ascending=False)
    .head(10)
    .round(0)
)
st.dataframe(top, use_container_width=True)

with st.expander("Raw data"):
    st.dataframe(f.head(500))
