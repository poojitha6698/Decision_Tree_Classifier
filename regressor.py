import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Decision Tree Regressor", layout="centered")

st.title("Decision Tree Regression")
st.write("Decision Tree regression using the Diabetes dataset or a custom CSV file")

# Sidebar settings
st.sidebar.header("Model Hyperparameters")

max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
criterion = st.sidebar.selectbox("Criterion", ["squared_error", "friedman_mse"])

# Dataset selection
data_option = st.radio(
    "Choose Dataset:",
    ["Diabetes Dataset (Default)", "Upload CSV"]
)

if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.warning("Please upload a CSV file")
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

else:
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target

    st.subheader("Diabetes Dataset Preview")
    st.dataframe(df.head())

    X = df.drop(columns=["target"])
    y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeRegressor(
    max_depth=max_depth,
    criterion=criterion,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
st.subheader("Model Performance")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Decision Tree Visualization
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(16, 9))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    ax=ax
)

st.pyplot(fig)

st.success("Decision Tree Regressor trained and visualized successfully")
