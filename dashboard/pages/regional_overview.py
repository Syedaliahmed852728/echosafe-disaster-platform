"""Regional command overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.utils.dashboard_ui import (
    hero, metric_card, load_regions, load_predictions, load_alerts, merge_regions_with_predictions,
    render_map, render_risk_legend, prediction_status_note, style_dataframe, RISK_ORDER,
)


def _overall_region_table() -> pd.DataFrame:
    regions = load_regions()
    preds = load_predictions()
    alerts = load_alerts()
    if preds.empty:
        table = regions.copy()
        table["overall_risk"] = "Pending"
        table["active_alerts"] = 0
        return table

    data = preds.copy()
    data["_rank"] = data["risk_level"].map(RISK_ORDER).fillna(0)
    pivot = data.pivot_table(index="region", columns="disaster_type", values="risk_level", aggfunc="first").reset_index()
    max_risk = data.sort_values(["region", "_rank"], ascending=[True, False]).groupby("region", as_index=False).head(1)[["region", "risk_level"]]
    max_risk = max_risk.rename(columns={"risk_level": "overall_risk"})
    table = regions.merge(pivot, on="region", how="left").merge(max_risk, on="region", how="left")
    table["overall_risk"] = table["overall_risk"].fillna("Pending")
    if not alerts.empty and "region" in alerts.columns:
        alert_counts = alerts.groupby("region").size().reset_index(name="active_alerts")
        table = table.merge(alert_counts, on="region", how="left")
    table["active_alerts"] = table.get("active_alerts", 0)
    table["active_alerts"] = table["active_alerts"].fillna(0).astype(int)
    return table


def render_regional_overview() -> None:
    hero(
        "Regional Command Overview",
        "One national operating picture for monitored regions, combining heatwave, flood, hailstorm, earthquake monitoring, and active alerts.",
        "Command Map",
    )

    table = _overall_region_table()
    pred_df = load_predictions()
    alerts_df = load_alerts()

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Monitored Regions", str(len(table)), "pilot coverage", "#38bdf8")
    with b:
        metric_card("Prediction Records", str(len(pred_df)) if not pred_df.empty else "0", "latest batch output", "#22c55e")
    with c:
        metric_card("Active Alerts", str(len(alerts_df)) if not alerts_df.empty else "0", "high / critical", "#ef4444")
    with d:
        high = 0 if pred_df.empty else int(pred_df["risk_level"].isin(["High", "Critical"]).sum())
        metric_card("High Priority", str(high), "records needing action", "#fb7185")

    st.markdown("<div class='section-title'>Combined Regional Map</div>", unsafe_allow_html=True)
    render_risk_legend()
    map_df = table.rename(columns={"overall_risk": "risk_level"})
    render_map(map_df, tooltip_cols=["region", "province", "risk_level", "active_alerts"], height=505)

    prediction_status_note()

    st.markdown("<div class='section-title'>Regional Risk Matrix</div>", unsafe_allow_html=True)
    display_cols = [c for c in table.columns if c not in ["latitude", "longitude"]]
    st.dataframe(style_dataframe(table[display_cols], ["overall_risk"] + [c for c in table.columns if "Risk" in c or "Monitoring" in c]), use_container_width=True, hide_index=True)
