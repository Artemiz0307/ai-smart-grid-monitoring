import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

st.set_page_config(page_title="AI Smart Grid Monitoring", layout="wide")

st.title("⚡ AI Smart Grid Monitoring System")

page = st.sidebar.selectbox(
    "Navigation",
    ["Demand Monitoring", "Forecasting", "Anomaly Detection", "Model Performance"]
)

df = pd.read_csv("data/PJME_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")


import numpy as np
import pickle

# Load models
autoencoder = tf.keras.models.load_model(
    "models/lstm_autoencoder_model.h5",
    compile=False
)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare data
data = df["PJME_MW"].values.reshape(-1,1)
data_scaled = scaler.transform(data)

window = 30

X = []
for i in range(window, len(data_scaled)):
    X.append(data_scaled[i-window:i])

X = np.array(X)

# Reconstruction
X_pred = autoencoder.predict(X)

reconstruction_error = np.mean(
    np.abs(X_pred - X),
    axis=(1,2)
)

threshold = np.mean(reconstruction_error) + 3*np.std(reconstruction_error)

anomalies = reconstruction_error > threshold

anomaly_indices = np.where(anomalies)[0]
anomaly_timestamps = df.index[anomaly_indices + window]

# ------------------------------
# PAGE 1 — Demand Monitoring
# ------------------------------
if page == "Demand Monitoring":

    st.header("Electricity Demand Monitoring")

    demand = df["PJME_MW"]

    # ---- CONTROL PANEL METRICS ----
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Demand",
        f"{int(demand.mean())} MW"
    )

    col2.metric(
        "Peak Demand",
        f"{int(demand.max())} MW"
    )

    col3.metric(
        "Minimum Demand",
        f"{int(demand.min())} MW"
    )

    col4.metric(
        "Total Records",
        len(demand)
    )

    st.subheader("System Status")

    # Calculate anomalies count
    total_anomalies = len(anomaly_indices)

    # Last anomaly
    last_anomaly_time = anomaly_timestamps[-1] if len(anomaly_timestamps) > 0 else "None"

    col1, col2, col3 = st.columns(3)

    col1.success("🟢 System Status: Normal")

    col2.metric(
        "Total Anomalies Detected",
        total_anomalies
    )

    col3.write(f"Last anomaly detected: {last_anomaly_time}")

    st.subheader("Electricity Demand Trend")

    st.line_chart(demand)

    st.line_chart(df["PJME_MW"])


# ------------------------------
# PAGE 2 — Forecasting
# ------------------------------
elif page == "Forecasting":

    st.header("Electricity Demand Forecast (LSTM Model)")

    import numpy as np
    import pickle
    import matplotlib.pyplot as plt

    # Load model
    lstm_model = tf.keras.models.load_model(
        "models/lstm_forecasting_model.h5",
        compile=False
    )

    # Load scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    data = df["PJME_MW"].values.reshape(-1,1)
    data_scaled = scaler.transform(data)

    window = 30

    X = []
    for i in range(window, len(data_scaled)):
        X.append(data_scaled[i-window:i])

    X = np.array(X)

    # Predict
    predictions = lstm_model.predict(X)

    predictions = scaler.inverse_transform(predictions)

    actual = data[window:]

    # Plot
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(actual[:1000], label="Actual Demand")
    ax.plot(predictions[:1000], label="LSTM Forecast")

    ax.set_title("Electricity Demand Forecast")
    ax.legend()

    st.pyplot(fig)





    st.subheader("Next 24 Hour Demand Forecast")

    # Take the last 30 timesteps
    last_window = data_scaled[-window:]

    future_predictions = []

    current_window = last_window.copy()

    for _ in range(24):

        pred = lstm_model.predict(current_window.reshape(1, window, 1))
        future_predictions.append(pred[0][0])

        # slide the window
        current_window = np.append(current_window[1:], pred)

    future_predictions = np.array(future_predictions).reshape(-1,1)

    future_predictions = scaler.inverse_transform(future_predictions)

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(range(24), future_predictions, marker="o")

    ax2.set_title("Predicted Electricity Demand (Next 24 Hours)")
    ax2.set_xlabel("Hours Ahead")
    ax2.set_ylabel("MW")

    st.pyplot(fig2)


    st.subheader("🔮 Predict Next Electricity Demand")

if st.button("Predict Next Demand"):

    # take last 30 values
    last_window = data_scaled[-30:]

    # reshape for LSTM
    last_window = last_window.reshape(1, 30, 1)

    prediction = lstm_model.predict(last_window)

    # convert back to MW
    prediction = scaler.inverse_transform(prediction)

    st.success(f"Predicted Next Demand: {int(prediction[0][0])} MW")

# ------------------------------
# PAGE 3 — Anomaly Detection
# ------------------------------
elif page == "Anomaly Detection":

    st.header("AI Detected Grid Anomalies")

    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    # Load models
    autoencoder = tf.keras.models.load_model(
    "models/lstm_autoencoder_model.h5",
    compile=False
)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Prepare data
    data = df["PJME_MW"].values.reshape(-1,1)
    data_scaled = scaler.transform(data)

    window = 30

    X = []
    for i in range(window, len(data_scaled)):
        X.append(data_scaled[i-window:i])

    X = np.array(X)

    # Reconstruction
    X_pred = autoencoder.predict(X)

    reconstruction_error = np.mean(
        np.abs(X_pred - X),
        axis=(1,2)
    )

    threshold = np.mean(reconstruction_error) + 3*np.std(reconstruction_error)

    anomalies = reconstruction_error > threshold

    anomaly_indices = np.where(anomalies)[0]
    anomaly_timestamps = df.index[anomaly_indices + window]

    # Plot
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(df.index, df["PJME_MW"], label="Electricity Demand")

    ax.scatter(
        anomaly_timestamps,
        df.loc[anomaly_timestamps]["PJME_MW"],
        color="red",
        label="Detected Anomalies",
        s=10
    )

    ax.set_title("Smart Grid Anomaly Detection")
    ax.legend()

    st.pyplot(fig)

    st.subheader("⚠ Smart Grid Alerts")

    anomaly_data = pd.DataFrame({
        "Timestamp": anomaly_timestamps,
        "Demand_MW": df.loc[anomaly_timestamps]["PJME_MW"]
    })

    # show latest anomalies
    latest_anomalies = anomaly_data.tail(10)

    st.dataframe(latest_anomalies)


    if len(anomaly_data) > 0:

        latest_event = anomaly_data.iloc[-1]

        st.error(
            f"""
    ⚠ ALERT: Abnormal electricity consumption detected  

    Time: {latest_event['Timestamp']}  
    Demand: {latest_event['Demand_MW']} MW
    """
        )
# ------------------------------
# PAGE 4 — Model Performance
# ------------------------------
elif page == "Model Performance":

    st.header("Model Comparison")

    results = pd.DataFrame({
        "Model": ["SARIMA", "LSTM"],
        "RMSE": [5388, 2194],
        "MAE": [4092, 1682]
    })

    st.table(results)