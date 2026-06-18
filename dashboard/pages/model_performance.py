"""Model Performance Page."""

import streamlit as st
import json
import pandas as pd
from backend.config.settings import SETTINGS


def render_model_performance():
    st.title("📈 Model Performance")
    st.markdown("Evaluation metrics for all trained ML models.")
    st.markdown("---")

    # Time-based split explanation
    with st.expander("ℹ️ Training/Test Split Strategy", expanded=False):
        st.markdown("""
        **Time-based split is used (not random) because weather is temporal.**
        - **Training:** 2010-01-01 to 2022-12-31 (13 years)
        - **Validation:** 2023-01-01 to 2024-12-31 (2 years)
        - **Testing:** 2025-01-01 to 2025-12-31 (1 year)
        """)

    for disaster in ["flood", "heatwave", "hailstorm"]:
        st.subheader(
            f"{'🌊' if disaster == 'flood' else '🌡️' if disaster == 'heatwave' else '⛈️'} {disaster.title()} Risk Model"
        )

        try:
            with open(
                SETTINGS.project_root
                / "reports"
                / "model_metrics"
                / f"{disaster}_risk_metrics.json"
            ) as f:
                metrics = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.2%}")
            c3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.2%}")

            try:
                cm = pd.read_csv(
                    SETTINGS.project_root
                    / "reports"
                    / "model_metrics"
                    / f"{disaster}_risk_confusion_matrix.csv",
                    index_col=0,
                )
                st.markdown("**Confusion Matrix:**")
                st.dataframe(cm, use_container_width=True)
            except FileNotFoundError:
                pass

        except FileNotFoundError:
            st.info(f"{disaster.title()} model metrics not available yet.")

        st.markdown("---")

    st.subheader("🌍 Earthquake Module")
    st.markdown("""
    - **Method:** Rule-based severity classification (NO ML)
    - **Input:** Magnitude + Depth
    - **Output:** Low / Medium / High / Critical severity
    - **Note:** This module does NOT predict earthquake occurrence. It only classifies severity of detected events.
    """)
