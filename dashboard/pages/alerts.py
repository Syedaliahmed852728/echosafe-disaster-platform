"""Alerts page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.utils.dashboard_ui import hero, metric_card, load_alerts, load_regions, render_map, render_risk_legend, style_dataframe, risk_to_color


def render_alerts_page() -> None:
    hero(
        "Active Alerts Center",
        "High and critical disaster-risk records consolidated into one response queue for monitored regions.",
        "Alert Operations",
    )

    alerts = load_alerts()
    regions = load_regions()

    if alerts.empty:
        st.markdown(
            """
            <div class='soft-note'>
            No generated alert file is available yet, or there are currently no high/critical alerts. Run
            <b>python pipelines/run_batch_predictions.py</b> after model artifacts are ready.
            </div>
            """,
            unsafe_allow_html=True,
        )
        neutral = regions.copy()
        neutral["risk_level"] = "Pending"
        neutral["status"] = "No generated alerts"
        render_risk_legend()
        render_map(neutral, tooltip_cols=["region", "province", "status"], height=440)
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Alerts", str(len(alerts)), "active queue", "#ef4444")
    with c2:
        metric_card("Critical", str(int((alerts["risk_level"] == "Critical").sum())), "urgent", "#ef4444")
    with c3:
        metric_card("High", str(int((alerts["risk_level"] == "High").sum())), "priority", "#fb7185")
    with c4:
        metric_card("Regions", str(alerts["region"].nunique() if "region" in alerts.columns else 0), "affected monitored areas", "#38bdf8")

    f1, f2, f3 = st.columns(3)
    with f1:
        region_filter = st.multiselect("Region", sorted(alerts["region"].dropna().unique().tolist()) if "region" in alerts.columns else [])
    with f2:
        type_filter = st.multiselect("Disaster Type", sorted(alerts["disaster_type"].dropna().unique().tolist()) if "disaster_type" in alerts.columns else [])
    with f3:
        level_filter = st.multiselect("Severity", ["High", "Critical"], default=["High", "Critical"])

    filtered = alerts.copy()
    if region_filter:
        filtered = filtered[filtered["region"].isin(region_filter)]
    if type_filter:
        filtered = filtered[filtered["disaster_type"].isin(type_filter)]
    if level_filter and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(level_filter)]

    map_df = regions.merge(filtered[["region", "risk_level", "disaster_type", "message"]].drop_duplicates("region"), on="region", how="inner") if "region" in filtered.columns else pd.DataFrame()
    st.markdown("<div class='section-title'>Alert Map</div>", unsafe_allow_html=True)
    render_risk_legend()
    render_map(map_df, tooltip_cols=["region", "province", "risk_level", "disaster_type", "message"], height=445)

    st.markdown("<div class='section-title'>Alert Queue</div>", unsafe_allow_html=True)
    for _, row in filtered.head(10).iterrows():
        level = row.get("risk_level", "High")
        color = risk_to_color(level)
        st.markdown(
            f"""
            <div class='glass-card' style='border-left:5px solid {color}; margin-bottom:.75rem;'>
              <div style='display:flex;align-items:center;justify-content:space-between;gap:1rem;'>
                <div style='font-weight:900;font-size:1.05rem;'>{row.get('disaster_type','Alert')} · {row.get('region','Unknown')}</div>
                <div style='color:{color};font-weight:900;'>{level}</div>
              </div>
              <div style='color:#94a3b8;font-size:.83rem;margin:.25rem 0;'>{row.get('timestamp','')}</div>
              <div style='color:#e2e8f0;'>{row.get('message','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    show_cols = [c for c in ["timestamp", "region", "disaster_type", "risk_level", "message", "status"] if c in filtered.columns]
    st.dataframe(style_dataframe(filtered[show_cols]), use_container_width=True, hide_index=True)
    st.download_button("Download Filtered Alerts CSV", filtered.to_csv(index=False), "filtered_alerts.csv", "text/csv", use_container_width=True)
