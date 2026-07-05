
import streamlit as st
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🚀", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🚀 Food Delivery Time Predictor")
st.info("Predict the estimated food delivery time using a Random Forest Machine Learning model.")

with st.sidebar:
    st.header("📌 Project")
    st.write("*Developer:* Manav Sharma")
    st.write("*Course:* B.Sc. Data Science & AI")
    st.metric("Algorithm", "Random Forest")
    st.metric("Features", "9")
    st.metric("Dataset Rows", "45,593")

col1, col2 = st.columns(2)
with col1:
    distance = st.number_input("📍 Distance (km)", 0.0, 100.0, 1.0, 0.1)
with col2:
    rating = st.number_input("⭐ Delivery Rating", 0.0, 5.0, 4.5, 0.1)

order = st.selectbox("🍔 Order Type", ["Drinks", "Meal", "Snack"])
vehicle = st.selectbox("🛵 Vehicle Type", ["electric_scooter", "motorcycle", "scooter"])

c1, c2 = st.columns(2)

# ================== MAP ==================
st.subheader("📍 Delivery Map")

restaurant_lat = st.number_input("Restaurant Latitude", value=19.0760)
restaurant_lon = st.number_input("Restaurant Longitude", value=72.8777)

delivery_lat = st.number_input("Delivery Latitude", value=19.1150)
delivery_lon = st.number_input("Delivery Longitude", value=72.9080)

map_data = pd.DataFrame({
    'lat': [restaurant_lat, delivery_lat],
    'lon': [restaurant_lon, delivery_lon]
})

st.map(map_data)

predict = c1.button("🚀 Predict", use_container_width=True)
clear = c2.button("🗑️ Clear History", use_container_width=True)

if clear:
    st.session_state.history = []

# ================== PREDICTION ==================
if predict:
    data = [0] * 9
    data[0] = 25
    data[1] = rating

    if order == "Drinks":
        data[2] = 1
    elif order == "Meal":
        data[3] = 1
    else:
        data[4] = 1

    if vehicle == "electric_scooter":
        data[5] = 1
    elif vehicle == "motorcycle":
        data[6] = 1
    else:
        data[7] = 1

    data[8] = distance

    with st.spinner("Predicting..."):
        time.sleep(1)
        prediction = model.predict([data])[0]

    st.balloons()

    # ================== RESULT ==================
    st.subheader("📊 Prediction Summary")

    st.success(f"🚀 Estimated Delivery Time: {prediction:.2f} minutes")

    a, b = st.columns(2)
    a.metric("Estimated Time", f"{prediction:.2f} min")
    b.metric("Distance", f"{distance:.1f} km")

    st.write(f"⭐ Rating: *{rating}*")
    st.write(f"🍔 Order Type: *{order}*")
    st.write(f"🛵 Vehicle: *{vehicle}*")

    # ================== GRAPH ==================
    st.subheader("📊 Prediction Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Predicted Time"], [prediction])
    ax.set_ylabel("Minutes")
    ax.set_title("Delivery Time Prediction")

    st.pyplot(fig)

    # ================== STATUS ==================
    if prediction < 20:
        st.success("⚡ Fast Delivery")
    elif prediction < 40:
        st.warning("🟡 Normal Delivery")
    else:
        st.error("🔴 Delivery may take longer")

    st.progress(min(int(prediction), 100))

    # ================== HISTORY ==================
    st.session_state.history.append({
        "Distance (km)": distance,
        "Rating": rating,
        "Order": order,
        "Vehicle": vehicle,
        "Predicted Time (min)": round(float(prediction), 2)
    })

# ================== HISTORY TABLE ==================
st.divider()
st.subheader("📝 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Download History (CSV)",
        df.to_csv(index=False),
        "prediction_history.csv",
        "text/csv",
        use_container_width=True
    )
else:
    st.info("No predictions yet.")

# ================== ABOUT ==================
with st.expander("ℹ️ About this Project"):
    st.write("""
This application predicts food delivery time using a Random Forest Regression model.
It considers:
- Delivery Distance
- Delivery Rating
- Order Type
- Vehicle Type
""")

st.markdown("---")
st.caption("Developed by Manav Sharma • Python • Scikit-learn • Streamlit")
