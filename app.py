"""
Humanitarian Aid Intelligence Dashboard
Streamlit app — reads from Databricks Delta tables or local CSV fallback.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Humanitarian Aid Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME / STYLE ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #141929 100%);
        border: 1px solid #2a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #6b7db3;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: #e8edff;
    }
    .metric-delta-pos { color: #34d399; font-size: 13px; }
    .metric-delta-neg { color: #f87171; font-size: 13px; }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #3b82f6;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Anomaly badge */
    .anomaly-badge {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .normal-badge {
        background: #14532d;
        color: #86efac;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1224 !important;
        border-right: 1px solid #1e2d4a;
    }

    /* Plotly chart containers */
    .chart-container {
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 4px;
    }

    h1 { color: #e8edff !important; }
    h2 { color: #c7d2fe !important; }
    h3 { color: #a5b4fc !important; }
    p, li { color: #9ca3af !important; }
    .stMarkdown { color: #9ca3af !important; }
    label { color: #9ca3af !important; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADER ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    # from pyspark.sql import SparkSession
    from databricks.connect import DatabricksSession
    import os
    # host = os.environ.get("DATABRICKS_HOST")
    # cluster = os.environ.get("DATABRICKS_WORKSPACE_ID")
    # cluster = os.getenv("CLUSTER_ID")
    # token = os.getenv("TOKEN")
    # print("envs", host, cluster, token)

    # spark = DatabricksSession.builder \
    #     .host(host) \
    #     .clusterId(cluster) \
    #     .token(token) \
    #     .getOrCreate()

    # spark = SparkSession.builder \
    # .appName("humanitarian-ingestion") \
    # .config("spark.sql.adaptive.enabled", "true") \
    # .getOrCreate()
    # host = os.environ["DATABRICKS_HOST"]
    cluster_id = "0222-082255-oh0wkr3i-v2n"
    # print(host, cluster_id)

    # spark = DatabricksSession.builder \
    #     .host(host) \
    #     .clusterId(cluster_id) \
    #     .getOrCreate()  # no .token() — OAuth handles auth automatically
    spark = DatabricksSession.builder.clusterId(cluster_id).getOrCreate()

    features  = spark.table("humanitarian.features").toPandas()
    anomalies = spark.table("humanitarian.anomalies").toPandas()
    projects  = spark.table("humanitarian.projects").toPandas()
    contribs  = spark.table("humanitarian.contributions").toPandas()

    return features, anomalies, projects, contribs

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading humanitarian data..."):
    features, anomalies, projects, contribs = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Aid Intelligence")
    st.markdown("---")

    all_countries = sorted(features["country_code"].dropna().unique().tolist())
    all_clusters  = sorted(features["cluster_name"].dropna().unique().tolist())
    all_years     = sorted(features["year"].dropna().unique().tolist(), reverse=True)

    selected_countries = st.multiselect("Countries", all_countries,
                                        default=all_countries[:8])
    selected_clusters  = st.multiselect("Clusters / Sectors", all_clusters,
                                        default=all_clusters[:5])
    selected_years     = st.multiselect("Years", all_years,
                                        default=all_years[:2])

    st.markdown("---")
    show_anomalies_only = st.toggle("Show Anomalies Only", value=False)
    funding_gap_min = st.slider("Min Funding Gap %", 0, 100, 0)

    st.markdown("---")
    st.markdown("<div style='color:#6b7db3;font-size:11px;'>Built on Databricks · Delta Lake · MLflow</div>",
                unsafe_allow_html=True)

# ── FILTER DATA ───────────────────────────────────────────────────────────────
def apply_filters(df):
    mask = pd.Series([True] * len(df), index=df.index)
    if selected_countries and "country_code" in df.columns:
        mask &= df["country_code"].isin(selected_countries)
    if selected_clusters and "cluster_name" in df.columns:
        mask &= df["cluster_name"].isin(selected_clusters)
    if selected_years and "year" in df.columns:
        mask &= df["year"].isin(selected_years)
    if "funding_gap_pct" in df.columns:
        mask &= df["funding_gap_pct"].fillna(0) >= funding_gap_min
    return df[mask].copy()

feat_f   = apply_filters(features)
anom_f   = apply_filters(anomalies)
proj_f   = apply_filters(projects.rename(columns={"country_name": "country_code"}))
proj_f   = proj_f.rename(columns={"country_code": "country_name"})

if show_anomalies_only and "is_anomaly" in anom_f.columns:
    anom_f = anom_f[anom_f["is_anomaly"] == 1]

PLOTLY_THEME = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#9ca3af", family="Space Grotesk"),
    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 16px 0;'>
  <h1 style='font-size:36px;font-weight:700;margin:0;background:linear-gradient(90deg,#60a5fa,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    Humanitarian Aid Intelligence
  </h1>
  <p style='color:#6b7db3;margin:4px 0 0 0;font-size:15px;'>
    Anomaly detection · Funding gap analysis · Geo-visualization · Portfolio optimization
  </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🚨 Anomalies",
    "🗺️ Geo Map",
    "📈 Funding Flows",
    "⚖️ Portfolio",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    total_req  = feat_f["total_requirements_usd"].sum()
    total_fund = feat_f["total_funding_usd"].sum()
    total_gap  = feat_f["funding_gap_usd"].sum() if "funding_gap_usd" in feat_f.columns else total_req - total_fund
    avg_cov    = feat_f["funding_coverage_rate"].mean() * 100 if "funding_coverage_rate" in feat_f.columns else 0
    n_anomalies = anom_f["is_anomaly"].sum() if "is_anomaly" in anom_f.columns else 0
    total_inneed = feat_f["total_inneed"].sum() if "total_inneed" in feat_f.columns else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col_obj, label, value, delta=None, delta_pos=True):
        delta_html = ""
        if delta:
            cls = "metric-delta-pos" if delta_pos else "metric-delta-neg"
            delta_html = f"<div class='{cls}'>{delta}</div>"
        col_obj.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{value}</div>
          {delta_html}
        </div>""", unsafe_allow_html=True)

    kpi(k1, "Total Requirements", f"${total_req/1e9:.1f}B")
    kpi(k2, "Total Funded", f"${total_fund/1e9:.1f}B")
    kpi(k3, "Funding Gap", f"${total_gap/1e9:.1f}B", delta_pos=False)
    kpi(k4, "Avg Coverage Rate", f"{avg_cov:.1f}%")
    kpi(k5, "People In Need", f"{total_inneed/1e6:.1f}M")
    kpi(k6, "Anomalies Flagged", f"{n_anomalies:,}", delta_pos=False)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Funding Gap by Country</div>", unsafe_allow_html=True)
        country_agg = feat_f.groupby("country_code").agg(
            requirements=("total_requirements_usd", "sum"),
            funding=("total_funding_usd", "sum"),
        ).reset_index()
        country_agg["gap_pct"] = (country_agg["requirements"] - country_agg["funding"]) / (country_agg["requirements"] + 1e-9) * 100
        country_agg = country_agg.sort_values("gap_pct", ascending=False).head(20)

        fig = px.bar(
            country_agg, x="country_code", y="gap_pct",
            color="gap_pct",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            labels={"gap_pct": "Funding Gap %", "country_code": "Country"},
        )
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Funding Coverage by Cluster</div>", unsafe_allow_html=True)
        cluster_agg = feat_f.groupby("cluster_name").agg(
            requirements=("total_requirements_usd", "sum"),
            funding=("total_funding_usd", "sum"),
        ).reset_index()
        cluster_agg["coverage"] = cluster_agg["funding"] / (cluster_agg["requirements"] + 1e-9) * 100
        cluster_agg = cluster_agg.sort_values("coverage")

        fig2 = px.bar(
            cluster_agg, x="coverage", y="cluster_name",
            orientation="h",
            color="coverage",
            color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
            labels={"coverage": "Coverage %", "cluster_name": "Cluster"},
        )
        fig2.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                           margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Trend line
    st.markdown("<div class='section-header'>Funding Trends Over Time</div>", unsafe_allow_html=True)
    trend = feat_f.groupby("year").agg(
        requirements=("total_requirements_usd", "sum"),
        funding=("total_funding_usd", "sum"),
    ).reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=trend["year"], y=trend["requirements"]/1e9,
                              name="Requirements", line=dict(color="#3b82f6", width=2)))
    fig3.add_trace(go.Scatter(x=trend["year"], y=trend["funding"]/1e9,
                              name="Funded", line=dict(color="#22c55e", width=2),
                              fill="tonexty", fillcolor="rgba(34,197,94,0.1)"))
    fig3.update_layout(**PLOTLY_THEME, height=280, yaxis_title="USD Billion",
                       margin=dict(l=0,r=0,t=10,b=0),
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🚨 Anomalous Aid Allocations (Isolation Forest)</div>",
                unsafe_allow_html=True)

    if "is_anomaly" not in anom_f.columns:
        st.warning("Anomaly data not available. Run Notebook 03 first.")
    else:
        a1, a2 = st.columns([2, 1])
        with a1:
            # Scatter: anomaly score vs funding gap
            fig_a = px.scatter(
                anom_f.dropna(subset=["anomaly_score","funding_gap_pct"]),
                x="funding_gap_pct", y="anomaly_score",
                color="is_anomaly",
                color_continuous_scale=["#22c55e","#ef4444"],
                hover_data=["country_code","cluster_name","year"],
                labels={"funding_gap_pct":"Funding Gap %","anomaly_score":"Anomaly Score"},
                title="Anomaly Score vs Funding Gap",
            )
            fig_a.update_layout(**PLOTLY_THEME, height=380,
                                coloraxis_showscale=False,
                                margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_a, use_container_width=True)

        with a2:
            st.markdown("**Cluster Distribution**")
            if "cluster_id" in anom_f.columns:
                cluster_dist = anom_f.groupby(["cluster_id","is_anomaly"]).size().reset_index(name="count")
                fig_pie = px.pie(
                    anom_f[anom_f["is_anomaly"]==1],
                    names="cluster_name" if "cluster_name" in anom_f.columns else "cluster_id",
                    title="Anomalies by Sector",
                    color_discrete_sequence=px.colors.sequential.Reds_r,
                )
                fig_pie.update_layout(**PLOTLY_THEME, height=380,
                                      margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

        # Top anomalies table
        st.markdown("<div class='section-header'>Top Flagged Allocations</div>",
                    unsafe_allow_html=True)
        top20 = anom_f[anom_f["is_anomaly"]==1].sort_values(
            "anomaly_score", ascending=False
        ).head(20)[[
            "country_code","cluster_name","year","appeal_name",
            "total_requirements_usd","total_funding_usd","funding_gap_pct","anomaly_score"
        ]].copy()
        top20["total_requirements_usd"] = top20["total_requirements_usd"].apply(lambda x: f"${x:,.0f}")
        top20["total_funding_usd"]       = top20["total_funding_usd"].apply(lambda x: f"${x:,.0f}")
        top20["funding_gap_pct"]         = top20["funding_gap_pct"].apply(lambda x: f"{x:.1f}%")
        top20["anomaly_score"]           = top20["anomaly_score"].apply(lambda x: f"{x:.3f}")
        top20.columns = ["Country","Cluster","Year","Appeal","Requirements","Funded","Gap %","Score"]
        st.dataframe(top20, use_container_width=True, hide_index=True)

        # Radar chart for worst anomaly
        st.markdown("<div class='section-header'>Feature Radar — Worst Anomaly</div>",
                    unsafe_allow_html=True)
        ML_FEATURES_RADAR = [
            "funding_coverage_rate","funding_gap_pct","beneficiary_to_funding_ratio",
            "need_to_requirements_ratio","targeting_coverage_rate","disaster_severity_score",
        ]
        radar_feats = [f for f in ML_FEATURES_RADAR if f in anom_f.columns]
        if radar_feats and len(anom_f[anom_f["is_anomaly"]==1]) > 0:
            worst   = anom_f[anom_f["is_anomaly"]==1].sort_values("anomaly_score",ascending=False).iloc[0]
            avg_all = anom_f[radar_feats].mean()
            from sklearn.preprocessing import MinMaxScaler
            combined = pd.DataFrame([worst[radar_feats], avg_all], index=["Worst Anomaly","Average"])
            combined_norm = pd.DataFrame(
                MinMaxScaler().fit_transform(combined.T),
                index=radar_feats, columns=combined.index
            ).T

            fig_radar = go.Figure()
            for idx, row in combined_norm.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row.tolist() + [row.iloc[0]],
                    theta=radar_feats + [radar_feats[0]],
                    name=idx,
                    fill="toself" if idx=="Worst Anomaly" else None,
                    line=dict(color="#ef4444" if idx=="Worst Anomaly" else "#3b82f6"),
                ))
            fig_radar.update_layout(
                **PLOTLY_THEME,
                polar=dict(
                    bgcolor="#111827",
                    radialaxis=dict(visible=True, color="#374151"),
                    angularaxis=dict(color="#374151"),
                ),
                height=380, margin=dict(l=40,r=40,t=10,b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: GEO MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🗺️ Global Funding Gap Map</div>",
                unsafe_allow_html=True)

    map_metric = st.selectbox(
        "Map Metric",
        ["funding_gap_pct","funding_coverage_rate","total_funding_usd",
         "total_requirements_usd","disaster_severity_score","total_inneed"],
        index=0
    )

    geo_df = feat_f.groupby("country_code")[map_metric].mean().reset_index()
    geo_df.columns = ["country_code", "value"]

    fig_map = px.choropleth(
        geo_df,
        locations="country_code",
        color="value",
        color_continuous_scale=["#22c55e","#eab308","#ef4444"] if "gap" in map_metric
            else ["#ef4444","#eab308","#22c55e"],
        labels={"value": map_metric.replace("_"," ").title()},
        projection="natural earth",
    )
    fig_map.update_layout(
        **PLOTLY_THEME,
        geo=dict(
            bgcolor="#111827",
            landcolor="#1a2035",
            oceancolor="#0a0e1a",
            showocean=True,
            showframe=False,
            showcountries=True,
            countrycolor="#2a3a5c",
        ),
        height=500,
        margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_colorbar=dict(
            bgcolor="#111827",
            tickfont=dict(color="#9ca3af"),
            title=dict(font=dict(color="#9ca3af")),
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Project geo scatter
    if "latitude" in proj_f.columns and proj_f["latitude"].notna().sum() > 0:
        st.markdown("<div class='section-header'>Project Locations (CBPF)</div>",
                    unsafe_allow_html=True)
        proj_geo = proj_f.dropna(subset=["latitude","longitude"]).copy()
        fig_geo = px.scatter_geo(
            proj_geo.head(1000),
            lat="latitude", lon="longitude",
            color="cluster" if "cluster" in proj_geo.columns else None,
            size="budget" if "budget" in proj_geo.columns else None,
            size_max=15,
            hover_data=["org_name","cluster","budget"] if "budget" in proj_geo.columns else None,
            projection="natural earth",
        )
        fig_geo.update_layout(
            **PLOTLY_THEME,
            geo=dict(bgcolor="#111827", landcolor="#1a2035",
                     oceancolor="#0a0e1a", showocean=True,
                     countrycolor="#2a3a5c"),
            height=450, margin=dict(l=0,r=0,t=0,b=0),
        )
        st.plotly_chart(fig_geo, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: FUNDING FLOWS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>📈 Funding Flow Analysis</div>",
                unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        # Requirements vs Funded bubble
        bubble_df = feat_f.groupby("country_code").agg(
            requirements=("total_requirements_usd","sum"),
            funding=("total_funding_usd","sum"),
            inneed=("total_inneed","sum"),
        ).reset_index()
        fig_bub = px.scatter(
            bubble_df,
            x="requirements", y="funding",
            size="inneed", color="country_code",
            hover_data=["country_code"],
            labels={"requirements":"Requirements (USD)","funding":"Funded (USD)"},
            title="Requirements vs Funding by Country",
            size_max=50,
        )
        # diagonal line = fully funded
        max_val = max(bubble_df["requirements"].max(), bubble_df["funding"].max())
        fig_bub.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="100% funded",
            line=dict(color="#6b7db3", dash="dash", width=1),
        ))
        fig_bub.update_layout(**PLOTLY_THEME, height=380,
                              showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_bub, use_container_width=True)

    with f2:
        # Sector funding efficiency heatmap
        if "sector_funding_efficiency_pct" in feat_f.columns:
            heat_df = feat_f.groupby(["country_code","cluster_name"])["funding_coverage_rate"].mean().reset_index()
            heat_pivot = heat_df.pivot(index="country_code", columns="cluster_name",
                                       values="funding_coverage_rate").fillna(0)
            heat_pivot = heat_pivot.loc[heat_pivot.mean(axis=1).sort_values().index[:20]]

            fig_heat = px.imshow(
                heat_pivot,
                color_continuous_scale=["#1e3a2f","#22c55e"],
                labels=dict(color="Coverage Rate"),
                title="Funding Coverage Heatmap (Country × Sector)",
                aspect="auto",
            )
            fig_heat.update_layout(**PLOTLY_THEME, height=380,
                                   margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

    # Donor contributions
    st.markdown("<div class='section-header'>Donor Contributions (CBPF)</div>",
                unsafe_allow_html=True)
    if len(contribs) > 0:
        contrib_filt = contribs.copy()
        if selected_years:
            contrib_filt = contrib_filt[contrib_filt["year"].isin(selected_years)]

        c3, c4 = st.columns(2)
        with c3:
            top_funds = contrib_filt.groupby("country_name")["total_paid_usd"].sum() \
                            .sort_values(ascending=False).head(15).reset_index()
            fig_don = px.bar(top_funds, x="total_paid_usd", y="country_name",
                             orientation="h",
                             color="total_paid_usd",
                             color_continuous_scale="Blues",
                             title="Top 15 Country Funds (Paid USD)")
            fig_don.update_layout(**PLOTLY_THEME, height=380, coloraxis_showscale=False,
                                  margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_don, use_container_width=True)

        with c4:
            pledge_vs_paid = contrib_filt.groupby("year").agg(
                pledged=("total_pledged_usd","sum"),
                paid=("total_paid_usd","sum"),
            ).reset_index()
            fig_pp = go.Figure()
            fig_pp.add_trace(go.Bar(x=pledge_vs_paid["year"], y=pledge_vs_paid["pledged"]/1e6,
                                    name="Pledged", marker_color="#3b82f6"))
            fig_pp.add_trace(go.Bar(x=pledge_vs_paid["year"], y=pledge_vs_paid["paid"]/1e6,
                                    name="Paid", marker_color="#22c55e"))
            fig_pp.update_layout(**PLOTLY_THEME, barmode="group",
                                 title="Pledged vs Paid (M USD)", height=380,
                                 margin=dict(l=0,r=0,t=30,b=0),
                                 legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>⚖️ Portfolio Optimization — Maximize Impact per Dollar</div>",
                unsafe_allow_html=True)

    st.markdown("""
    **Framing**: Treat each country-cluster as an "asset". 
    Maximize total beneficiaries reached subject to a total budget constraint.
    """)

    total_available = st.slider(
        "Available Budget (USD Billion)",
        min_value=0.5, max_value=50.0,
        value=float(feat_f["total_funding_usd"].sum() / 1e9),
        step=0.5
    )
    budget_usd = total_available * 1e9

    try:
        from scipy.optimize import linprog

        opt_df = feat_f.groupby(["country_code","cluster_name"]).agg(
            requirements=("total_requirements_usd","sum"),
            targeted=("total_targeted","sum"),
        ).reset_index().dropna()
        opt_df = opt_df[opt_df["requirements"] > 0]

        # impact = beneficiaries per dollar
        opt_df["impact_per_dollar"] = opt_df["targeted"] / (opt_df["requirements"] + 1e-9)

        # LP: maximize sum(impact_i * x_i) s.t. sum(req_i * x_i) <= budget, 0<=x_i<=1
        c_obj    = -opt_df["impact_per_dollar"].values   # negate for minimization
        A_ub     = [opt_df["requirements"].values]
        b_ub     = [budget_usd]
        bounds   = [(0, 1)] * len(opt_df)

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            opt_df["allocation"] = result.x
            opt_df["allocated_usd"]        = opt_df["allocation"] * opt_df["requirements"]
            opt_df["expected_beneficiaries"] = opt_df["allocation"] * opt_df["targeted"]
            opt_df = opt_df.sort_values("expected_beneficiaries", ascending=False)

            total_beneficiaries = opt_df["expected_beneficiaries"].sum()
            total_allocated     = opt_df["allocated_usd"].sum()

            p1, p2, p3 = st.columns(3)
            kpi(p1, "Budget Allocated", f"${total_allocated/1e9:.2f}B")
            kpi(p2, "Expected Beneficiaries", f"{total_beneficiaries/1e6:.1f}M")
            kpi(p3, "Avg Impact/Dollar", f"{total_beneficiaries/max(total_allocated,1):.2f}")

            # Waterfall of top allocations
            top_alloc = opt_df[opt_df["allocation"] > 0.01].head(20)
            fig_opt = px.bar(
                top_alloc,
                x="expected_beneficiaries", y="country_code",
                color="cluster_name",
                orientation="h",
                title=f"Top Allocations — {total_beneficiaries/1e6:.1f}M beneficiaries reached",
                labels={"expected_beneficiaries":"Expected Beneficiaries","country_code":"Country"},
            )
            fig_opt.update_layout(**PLOTLY_THEME, height=480,
                                  margin=dict(l=0,r=0,t=30,b=0),
                                  legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_opt, use_container_width=True)

            # Full table
            st.markdown("<div class='section-header'>Full Allocation Table</div>",
                        unsafe_allow_html=True)
            display_opt = opt_df[opt_df["allocation"] > 0.01][[
                "country_code","cluster_name","requirements","allocated_usd",
                "allocation","expected_beneficiaries","impact_per_dollar"
            ]].copy()
            display_opt["requirements"]          = display_opt["requirements"].apply(lambda x: f"${x:,.0f}")
            display_opt["allocated_usd"]         = display_opt["allocated_usd"].apply(lambda x: f"${x:,.0f}")
            display_opt["allocation"]            = display_opt["allocation"].apply(lambda x: f"{x:.1%}")
            display_opt["expected_beneficiaries"] = display_opt["expected_beneficiaries"].apply(lambda x: f"{x:,.0f}")
            display_opt["impact_per_dollar"]     = display_opt["impact_per_dollar"].apply(lambda x: f"{x:.4f}")
            display_opt.columns = ["Country","Cluster","Requirements","Allocated",
                                   "Allocation %","Exp. Beneficiaries","Impact/Dollar"]
            st.dataframe(display_opt, use_container_width=True, hide_index=True)
        else:
            st.error(f"Optimization failed: {result.message}")

    except ImportError:
        st.error("scipy not installed. Run: pip install scipy")
    except Exception as e:
        st.error(f"Optimization error: {e}")
