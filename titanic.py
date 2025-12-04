import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# --- Page Config ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# --- Title & Description ---
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability based on historical data.")

# --- Load & Train Model (Runs instantly) ---
@st.cache_data
def load_and_train():
    df = sns.load_dataset('titanic')[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    X = df.drop('survived', axis=1)
    y = df['survived']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = load_and_train()

# --- User Inputs (Sidebar) ---
st.sidebar.header("User Input Features")

pclass = st.sidebar.selectbox("Class", [1, 2, 3], format_func=lambda x: f"{x}st Class")
sex = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 25)
fare = st.sidebar.slider("Ticket Fare ($)", 0, 500, 32)

# --- Prediction Logic ---
sex_num = 0 if sex == "Male" else 1
input_data = pd.DataFrame([[pclass, sex_num, age, fare]], columns=['pclass', 'sex', 'age', 'fare'])

prediction_prob = model.predict_proba(input_data)[0][1] * 100

# --- Display Result ---
st.subheader("Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction_prob > 50:
        st.success("You Survived! 🎉")
        st.image("https://media.giphy.com/media/9wLKh6ms5t9q4/giphy.gif")
    else:
        st.error("You Did Not Survive 🧊")
        st.image("https://media.giphy.com/media/OJw4CDqlr7JO/giphy.gif")

with col2:
    st.metric(label="Survival Probability", value=f"{prediction_prob:.1f}%")
    st.progress(int(prediction_prob))

# --- Data Viz ---
st.markdown("---")
st.subheader("How does the model decide?")
st.bar_chart(pd.DataFrame({'Importance': model.coef_[0]}, index=['Class', 'Sex', 'Age', 'Fare']))