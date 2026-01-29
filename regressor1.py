import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("California Housing Price Predictor")
st.write("Decision Tree Regressor using Streamlit")

st.divider()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("california_housing_test.csv")

df = load_data()

# Features & target
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Sidebar controls
st.sidebar.header("Model Settings")

max_depth = st.sidebar.slider(
    "Select Tree Depth",
    min_value=1,
    max_value=20,
    value=5
)

test_size = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=40,
    value=20
) / 100

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train model
model = DecisionTreeRegressor(
    max_depth=max_depth,
    criterion="squared_error",
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.subheader("Model Performance")
col1, col2 = st.columns(2)

col1.metric("Mean Squared Error", f"{mse:,.2f}")
col2.metric("R² Score", f"{r2:.3f}")

st.divider()

# User input section
st.subheader("Predict House Value")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(
        f"{col}",
        value=float(df[col].mean())
    )

input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Value: **${prediction:,.2f}**")

st.divider()

# Show dataset
with st.expander("View Dataset"):
    st.dataframe(df.head())
