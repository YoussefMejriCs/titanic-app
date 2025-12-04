import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- Page Config ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# --- 1. Load Data & Train Model ---
@st.cache_data
def load_and_train():
    # Load data directly from URL
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
    
    # Pre-calculate stats for the graphs BEFORE we drop columns
    survival_by_sex = df.groupby('sex')['survived'].mean() * 100
    survival_by_class = df.groupby('pclass')['survived'].mean() * 100
    
    # Prepare data for model
    df = df[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return model, survival_by_sex, survival_by_class

model, survival_by_sex, survival_by_class = load_and_train()

# --- 2. The Drama (Header) ---
st.title("🚢 The Titanic: Fate & Fortune")
st.markdown("""
*In the freezing waters of the North Atlantic, destiny was often decided by two things: 
**Chivalry** and **Currency**.*
""")

# --- 3. User Inputs (Sidebar) ---
st.sidebar.header("📝 Passenger Details")
pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class (Rich)" if x==1 else (f"{x}nd Class (Middle)" if x==2 else f"{x}rd Class (Poor)"))
sex = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
fare = st.sidebar.slider("Fare Paid ($)", 0, 500, 50)

# --- 4. Prediction ---
sex_num = 0 if sex == "Male" else 1
input_data = pd.DataFrame([[pclass, sex_num, age, fare]], columns=['pclass', 'sex', 'age', 'fare'])
probability = model.predict_proba(input_data)[0][1] * 100

# --- 5. The Reveal (Results) ---
st.markdown("---")
st.subheader("🔮 Will you survive?")

col1, col2 = st.columns([1, 2])

with col1:
    if probability > 50:
        st.success("## Yes, you survive!")
        st.balloons()
    else:
        st.error("## No, you do not.")

with col2:
    st.metric(label="Survival Probability", value=f"{probability:.1f}%")
    st.progress(int(probability))

# --- 6. The Story Behind the Data (Graphs) ---
st.markdown("---")
st.header("📊 The Story Behind the Model")

tab1, tab2 = st.tabs(["💔 The Sacrifice", "💰 The Price of Life"])

with tab1:
    st.subheader("Ladies First: The Ultimate Sacrifice")
    st.markdown("_History tells us that men stepped aside so women could live. The data proves it._")
    
    # Custom Data for Graph
    sex_chart_data = pd.DataFrame({
        "Survival Rate (%)": survival_by_sex.values,
        "Gender": ["Female", "Male"] 
    }).set_index("Gender")
    
    # Create Graph
    st.bar_chart(sex_chart_data, color="#ff4b4b") # Red for drama

with tab2:
    st.subheader("The Echo of Wealth in the Lifeboats")
    st.markdown("_Money couldn't buy happiness, but on that night, it often bought a seat in a lifeboat. The poor were left behind._")
    
    # Custom Data for Graph
    class_chart_data = pd.DataFrame({
        "Survival Rate (%)": survival_by_class.values,
        "Class": ["1st (Rich)", "2nd (Middle)", "3rd (Poor)"]
    }).set_index("Class")
    
    # Create Graph
    st.bar_chart(class_chart_data, color="#4caf50") # Green for money