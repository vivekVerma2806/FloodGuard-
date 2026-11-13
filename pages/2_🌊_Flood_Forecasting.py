"""FloodGuard AI — LSTM Flood Forecasting Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import (
    DATA_DIR, FEATURES, TARGET_REGRESSION, SEQUENCE_LENGTH,
    LSTM_UNITS, LSTM_EPOCHS, LSTM_BATCH_SIZE, TEST_SPLIT_RATIO,
)
from src.data_preprocessing import (
    load_data, clean_data, engineer_features,
    normalize_features, create_sequences, chronological_split,
)
from src.lstm_model import (
    build_lstm_model, train_lstm_model, predict_lstm,
    save_model, load_saved_model,
)

st.set_page_config(page_title="Flood Forecasting | FloodGuard AI", page_icon="🌊", layout="wide")


def kpi_card(title, value, delta=None):
    delta_html = ""
    if delta is not None:
        delta_html = f'<p style="margin:0; color:#666; font-size:0.9rem;">{delta}</p>'
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0;
                    background: #f8f9fa; text-align: center;">
            <h4 style="margin:0; color:#666;">{title}</h4>
            <h2 style="margin:0; color:#1f77b4;">{value}</h2>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


def _get_data():
    """Return processed DataFrame from session state or demo dataset."""
    if "processed_df" in st.session_state:
        return st.session_state["processed_df"]
    demo_path = os.path.join(DATA_DIR, "demo_dataset.csv")
    if os.path.exists(demo_path):
        df = load_data(demo_path)
        df = clean_data(df)
        df = engineer_features(df)
        return df
    return None


def main():
    st.title("🌊 Flood Forecasting (LSTM)")
    st.markdown(
        "Train a Long Short-Term Memory (LSTM) neural network to forecast river levels."
    )

    st.info(
        "⚠️ **Note:** This page uses demo / synthetic data for illustration purposes. "
        "Predictions should **not** be used for real-world flood response decisions.",
        icon="ℹ️",
    )
    st.markdown("---")

    df = _get_data()
    if df is None:
        st.warning("No data available. Please upload or load data from the **📊 Data Upload** page.")
        return

    # ── Station selector ─────────────────────────────────────────────────
    stations = sorted(df["station_name"].unique())
    selected_station = st.selectbox("Select monitoring station:", stations)
    sdf = df[df["station_name"] == selected_station].sort_values("date").reset_index(drop=True)

    st.markdown(f"**Records for {selected_station}:** {len(sdf):,}")

    # ── Ensure features exist ────────────────────────────────────────────
    available = [f for f in FEATURES if f in sdf.columns]
    if len(available) < 2:
        st.error("Insufficient feature columns. Please run feature engineering first.")
        return

    # ── Train / Load ─────────────────────────────────────────────────────
    col_train, col_load = st.columns(2)
    train_clicked = col_train.button("🚀 Train LSTM Model", type="primary")
    load_clicked = col_load.button("📂 Load Saved Model")

    if train_clicked:
        with st.spinner("Preparing data & training LSTM..."):
            try:
                train_df, test_df = chronological_split(sdf, TEST_SPLIT_RATIO)

                train_norm, scaler = normalize_features(train_df, available)
                # Transform test with same scaler
                test_norm = test_df.copy()
                test_norm[available] = scaler.transform(test_df[available])

                target_idx = available.index(TARGET_REGRESSION) if TARGET_REGRESSION in available else 0

                X_train, y_train = create_sequences(
                    train_norm[available].values,
                    train_norm[available].values[:, target_idx],
                    SEQUENCE_LENGTH,
                )
                X_test, y_test = create_sequences(
                    test_norm[available].values,
                    test_norm[available].values[:, target_idx],
                    SEQUENCE_LENGTH,
                )

                if len(X_train) == 0 or len(X_test) == 0:
                    st.error("Not enough data to create sequences. Try a shorter sequence length.")
                    return

                model = build_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    units=LSTM_UNITS,
                )

                progress = st.progress(0, text="Training…")
                history = train_lstm_model(
                    model, X_train, y_train, X_test, y_test,
                    epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
                )
                progress.progress(100, text="Training complete!")

                y_pred = predict_lstm(model, X_test)

                # Store for display
                st.session_state["lstm_history"] = history
                st.session_state["lstm_y_test"] = y_test
                st.session_state["lstm_y_pred"] = y_pred
                st.session_state["lstm_model"] = model
                st.session_state["lstm_scaler"] = scaler
                st.session_state["lstm_test_dates"] = test_df["date"].values[SEQUENCE_LENGTH:]

                st.success("✅ LSTM training complete!")
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

    if load_clicked:
        with st.spinner("Loading saved model..."):
            model = load_saved_model()
            if model is not None:
                st.session_state["lstm_model"] = model
                st.success("✅ Model loaded from disk.")
            else:
                st.warning("No saved model found. Train a model first.")

    # ── Results ──────────────────────────────────────────────────────────
    if "lstm_history" in st.session_state:
        st.markdown("---")
        st.subheader("📉 Training Loss Curve")
        history = st.session_state["lstm_history"]
        loss_data = pd.DataFrame({
            "Epoch": range(1, len(history.history["loss"]) + 1),
            "Train Loss": history.history["loss"],
            "Val Loss": history.history.get("val_loss", history.history["loss"]),
        })
        fig_loss = px.line(
            loss_data, x="Epoch", y=["Train Loss", "Val Loss"],
            title="Training & Validation Loss",
            template="plotly_white",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    if "lstm_y_test" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Actual vs Predicted River Level")

        y_test = st.session_state["lstm_y_test"]
        y_pred = st.session_state["lstm_y_pred"].flatten()
        dates = st.session_state.get("lstm_test_dates")

        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

        mc1, mc2 = st.columns(2)
        with mc1:
            kpi_card("MAE", f"{mae:.4f}", "Mean Absolute Error")
        with mc2:
            kpi_card("RMSE", f"{rmse:.4f}", "Root Mean Square Error")

        compare_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        if dates is not None and len(dates) == len(y_test):
            compare_df["Date"] = dates
            x_col = "Date"
        else:
            compare_df["Index"] = range(len(y_test))
            x_col = "Index"

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=compare_df[x_col], y=compare_df["Actual"],
                                       mode="lines", name="Actual", line=dict(color="#1f77b4")))
        fig_pred.add_trace(go.Scatter(x=compare_df[x_col], y=compare_df["Predicted"],
                                       mode="lines", name="Predicted", line=dict(color="#ff7f0e")))
        fig_pred.update_layout(
            title="Actual vs Predicted (Normalized River Level)",
            template="plotly_white",
            xaxis_title=x_col,
            yaxis_title="River Level (normalized)",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # ── Save Model ───────────────────────────────────────────────────────
    if "lstm_model" in st.session_state:
        st.markdown("---")
        if st.button("💾 Save Model to Disk"):
            try:
                save_model(st.session_state["lstm_model"])
                st.success("✅ Model saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")


try:
    main()
except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
