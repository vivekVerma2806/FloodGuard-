"""FloodGuard AI - Multivariate LSTM Flood Forecasting."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import (
    DATA_DIR,
    FEATURES,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_UNITS,
    SEQUENCE_LENGTH,
    TARGET_REGRESSION,
    TEST_SPLIT_RATIO,
)
from src.data_preprocessing import (
    chronological_split,
    clean_data,
    create_sequences,
    engineer_features,
    load_data,
    normalize_features,
)
from src.lstm_model import (
    build_lstm_model,
    load_saved_model,
    predict_lstm,
    save_model,
    train_lstm_model,
)
from src.ui_theme import (
    apply_theme,
    render_sidebar,
    editorial_header,
    kpi_card,
    editorial_callout,
    render_html,
    style_plotly_chart,
    TEXT_MUTED,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Flood Forecasting | FloodGuard AI",
    page_icon="☵",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar()


def _get_data():
    """Retrieve processed telemetry from session state or baseline file."""
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
    editorial_header(
        title="Multivariate LSTM Stage Forecasting",
        kicker="RECURRENT NEURAL ARCHITECTURE // TEMPORAL REGRESSION",
        subtitle=(
            "Train deep recurrent LSTM networks over sliding temporal windows to forecast next-day "
            "river gauge stages. Employs chronological evaluation to prevent lookahead bias."
        ),
    )

    df = _get_data()
    if df is None:
        editorial_callout(
            "Hydrological observations unavailable. Ingest records via the Data Upload module first.",
            label="TELEMETRY STATUS // MISSING",
            level="warning",
        )
        return

    # -------------------------------------------------------------------------
    # Sentinel Basin & Horizon Selection
    # -------------------------------------------------------------------------
    col_stn, col_seq, col_epochs = st.columns([2, 1, 1])
    with col_stn:
        stations = sorted(df["station_name"].unique())
        selected_station = st.selectbox("Sentinel Gauging Station", stations)
    with col_seq:
        seq_len = st.number_input("Sliding Window (Days)", min_value=3, max_value=30, value=SEQUENCE_LENGTH)
    with col_epochs:
        epochs = st.number_input("Max Epochs", min_value=2, max_value=100, value=min(LSTM_EPOCHS, 25))

    sdf = df[df["station_name"] == selected_station].sort_values("date").reset_index(drop=True)

    available_features = [f for f in FEATURES if f in sdf.columns]
    if len(available_features) < 2:
        editorial_callout(
            "Insufficient hydrological features detected. Run feature engineering in the Data Upload module.",
            label="FEATURE MATRIX DEFICIT",
            level="warning",
        )
        return

    render_html(
        f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; margin-bottom: 1.25rem;">
            // STATION OBSERVATIONS: {len(sdf):,} DAYS | ATTRIBUTES: {len(available_features)} | TARGET: {TARGET_REGRESSION}
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # Training Execution Controls
    # -------------------------------------------------------------------------
    c_train, c_load, _ = st.columns([1, 1, 2])
    with c_train:
        btn_train = st.button("Train LSTM Network")
    with c_load:
        btn_load = st.button("Load Serialized Model")

    if btn_train:
        with st.spinner("Partitioning sequences and optimizing weights..."):
            try:
                train_df, test_df = chronological_split(sdf, TEST_SPLIT_RATIO)

                train_norm, scaler = normalize_features(train_df, available_features)
                test_norm = test_df.copy()
                test_norm[available_features] = scaler.transform(test_df[available_features].values)

                target_idx = available_features.index(TARGET_REGRESSION) if TARGET_REGRESSION in available_features else 0

                X_train, y_train = create_sequences(
                    train_norm[available_features].values,
                    train_norm[available_features].values[:, target_idx],
                    int(seq_len),
                )
                X_test, y_test = create_sequences(
                    test_norm[available_features].values,
                    test_norm[available_features].values[:, target_idx],
                    int(seq_len),
                )

                if len(X_train) == 0 or len(X_test) == 0:
                    st.error("Insufficient samples for chosen window length.")
                    return

                model = build_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    units=LSTM_UNITS,
                )

                prog = st.progress(0, text="Gradient optimization in progress...")
                history = train_lstm_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    epochs=int(epochs),
                    batch_size=LSTM_BATCH_SIZE,
                )
                prog.progress(100, text="Convergence complete.")

                y_pred = predict_lstm(model, X_test)

                st.session_state["lstm_history"] = history
                st.session_state["lstm_y_test"] = y_test
                st.session_state["lstm_y_pred"] = y_pred
                st.session_state["lstm_model"] = model
                st.session_state["lstm_scaler"] = scaler
                st.session_state["lstm_test_dates"] = test_df["date"].values[int(seq_len):]

                editorial_callout(
                    f"LSTM training converged across {len(history.history['loss'])} epochs with EarlyStopping restoration.",
                    label="OPTIMIZATION STATUS // COMPLETE",
                    level="info",
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

    if btn_load:
        with st.spinner("Deserializing model artifact from disk..."):
            model = load_saved_model()
            if model is not None:
                st.session_state["lstm_model"] = model
                editorial_callout("Model weights successfully restored from models/lstm_flood_model.keras.", label="ARTIFACT RESTORED", level="info")
            else:
                editorial_callout("No serialized model artifact found in models/ directory.", label="FILE NOT FOUND", level="warning")

    # -------------------------------------------------------------------------
    # Evaluation Metrics & Visualizations
    # -------------------------------------------------------------------------
    if "lstm_y_test" in st.session_state:
        st.markdown("---")
        y_test = st.session_state["lstm_y_test"]
        y_pred = st.session_state["lstm_y_pred"].flatten()
        dates = st.session_state.get("lstm_test_dates")

        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

        render_html(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; margin-bottom: 0.5rem;">
                // OUT-OF-SAMPLE TEST VALIDATION SCORECARD
            </div>
            """
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Mean Absolute Error", f"{mae:.4f}", caption="Normalized datum scale")
        with c2:
            kpi_card("Root Mean Square Error", f"{rmse:.4f}", caption="Penalizes outlier deviations")
        with c3:
            kpi_card("Sequence Length", f"{int(seq_len)} Days", caption="Temporal lookback depth")
        with c4:
            kpi_card("Test Observations", f"{len(y_test):,}", caption="Unseen evaluation window")

        # Visual charts
        render_html("<div style='height: 16px;'></div>")
        tab_preds, tab_loss = st.tabs(["Hydrograph: Observed vs. Forecast", "Training & Validation Loss"])

        with tab_preds:
            compare_df = pd.DataFrame({"Observed": y_test, "Forecast": y_pred})
            if dates is not None and len(dates) == len(y_test):
                compare_df["Date"] = dates
                x_axis = "Date"
            else:
                compare_df["Sample Index"] = range(len(y_test))
                x_axis = "Sample Index"

            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=compare_df[x_axis],
                    y=compare_df["Observed"],
                    mode="lines",
                    name="Observed Gauge Stage",
                    line=dict(color="#171717", width=1.5),
                )
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=compare_df[x_axis],
                    y=compare_df["Forecast"],
                    mode="lines",
                    name="LSTM Forecast Lead",
                    line=dict(color="#B45309", width=2, dash="dash"),
                )
            )
            fig_pred.update_layout(
                title=f"Hydrometric Forecast Validation — {selected_station}",
                xaxis_title="Timeline",
                yaxis_title="Stage Elevation (Normalized)",
            )
            fig_pred = style_plotly_chart(fig_pred, height=420)
            st.plotly_chart(fig_pred, use_container_width=True)

        with tab_loss:
            if "lstm_history" in st.session_state:
                history = st.session_state["lstm_history"]
                epochs_range = range(1, len(history.history["loss"]) + 1)
                fig_loss = go.Figure()
                fig_loss.add_trace(
                    go.Scatter(
                        x=list(epochs_range),
                        y=history.history["loss"],
                        mode="lines",
                        name="Training Loss (MSE)",
                        line=dict(color="#171717", width=1.5),
                    )
                )
                val_losses = history.history.get("val_loss", history.history["loss"])
                fig_loss.add_trace(
                    go.Scatter(
                        x=list(epochs_range),
                        y=val_losses,
                        mode="lines",
                        name="Validation Loss (MSE)",
                        line=dict(color="#B45309", width=2),
                    )
                )
                fig_loss.update_layout(
                    title="Objective Function Convergence Profile",
                    xaxis_title="Optimization Epoch",
                    yaxis_title="Mean Squared Error",
                )
                fig_loss = style_plotly_chart(fig_loss, height=400)
                st.plotly_chart(fig_loss, use_container_width=True)

        # Artifact serialization
        st.markdown("---")
        col_save_btn, col_save_info = st.columns([1, 3])
        with col_save_btn:
            if st.button("Serialize Weights to Disk"):
                try:
                    save_model(st.session_state["lstm_model"])
                    editorial_callout("Saved weights to models/lstm_flood_model.keras.", label="PERSISTENCE CONFIRMED", level="info")
                except Exception as exc:
                    st.error(f"Serialization failed: {exc}")
        with col_save_info:
            st.caption("Persists compiled weights in native Keras 3 format for production deployment.")


try:
    main()
except Exception as exc:
    st.error(f"Runtime Exception in Flood Forecasting module: {exc}")
