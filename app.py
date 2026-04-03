import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Marketing Analytics", layout="wide")

# -------------------- TITLE --------------------
st.markdown("<h1 style='text-align:center;'>📊 Marketing Campaign Analytics</h1>", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
data = pd.read_csv("dataset/marketing_campaign.csv")
data = data.dropna()

# Encode categorical data
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Features & target
X = data.drop("Response", axis=1)
y = data["Response"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------- SESSION STATE --------------------
if "responses" not in st.session_state:
    st.session_state.responses = []

if "last_input" not in st.session_state:
    st.session_state.last_input = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ================= TOP SECTION =================
top_left, top_right = st.columns([1,1])

# -------- LEFT (INPUT) --------
with top_left:
    st.subheader("🧾 Enter Customer Details")

    age = st.slider("Age", 18, 60, 25)
    income = st.slider("Income", 20000, 80000, 30000)
    gender = st.selectbox("Gender", ["Male", "Female"])
    campaign = st.selectbox("Campaign Type", ["Email", "Social", "Ads"])
    previous = st.selectbox("Previous Response", [0, 1])

    gender_map = {"Male": 1, "Female": 0}
    campaign_map = {"Email": 0, "Social": 1, "Ads": 2}

    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Gender": [gender_map[gender]],
        "CampaignType": [campaign_map[campaign]],
        "PreviousResponse": [previous]
    })

    if st.button("🚀 Predict"):
        result = model.predict(input_data)[0]

        st.session_state.responses.append(result)
        st.session_state.last_result = result
        st.session_state.last_input = {
            "Age": age,
            "Income": income,
            "Gender": gender,
            "Campaign": campaign,
            "Previous": previous
        }

        # 🔥 AUTO SCROLL TO OUTPUT
        st.markdown(
            """
            <script>
                window.location.href = "#output_section";
            </script>
            """,
            unsafe_allow_html=True
        )

# -------- RIGHT (DATASET) --------
with top_right:
    st.subheader("📁 Dataset")

    st.dataframe(data.head(), use_container_width=True)

    st.write("📊 Rows:", data.shape[0])
    st.write("📊 Columns:", data.shape[1])

    if st.checkbox("Show Full Dataset"):
        st.dataframe(data, use_container_width=True)

# ================= BOTTOM SECTION =================
st.markdown("---")
bottom_left, bottom_right = st.columns([1,1])

# -------- LEFT (OUTPUT + GRAPH) --------
with bottom_left:
    # 🔥 ANCHOR POINT FOR AUTO SCROLL
    st.markdown("<div id='output_section'></div>", unsafe_allow_html=True)

    st.subheader("📊 Prediction Output")

    if st.session_state.last_result is not None:

        if st.session_state.last_result == 1:
            st.success("✅ Customer WILL respond")
        else:
            st.error("❌ Customer will NOT respond")

        # Pie Chart
        live_data = pd.Series(st.session_state.responses).value_counts()
        live_data = live_data.reindex([0, 1], fill_value=0)

        labels = ["Not Responded", "Responded"]

        fig, ax = plt.subplots(figsize=(4,4))

        ax.pie(
            live_data.values,
            labels=labels,
            autopct='%1.1f%%',
            colors=["#ff4b4b", "#2ecc71"],
            startangle=90
        )

        ax.axis('equal')

        st.pyplot(fig)

    else:
        st.info("No prediction yet")

# -------- RIGHT (REPORT) --------
with bottom_right:
    st.subheader("📄 Report")

    if st.session_state.last_result is not None:

        result_text = "WILL Respond" if st.session_state.last_result == 1 else "NOT Respond"

        st.markdown("### 👤 Customer Details")
        st.write(st.session_state.last_input)

        st.markdown("### 🤖 Prediction")
        st.write(f"Customer **{result_text}**")

        st.markdown("### 📊 Summary")
        st.write(f"Total Predictions: {len(st.session_state.responses)}")
        st.write(f"Responded: {st.session_state.responses.count(1)}")
        st.write(f"Not Responded: {st.session_state.responses.count(0)}")

        # Reset Button
        if st.button("🔄 Reset"):
            st.session_state.responses = []
            st.session_state.last_result = None
            st.session_state.last_input = None

    else:
        st.info("No report available")

# -------------------- FOOTER --------------------
st.markdown("---")
