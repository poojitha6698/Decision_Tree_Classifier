import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Decision Tree Classifier", layout="centered")

st.title("Decision Tree Classifier")
st.write("Decision Tree classification using the Iris dataset or a custom CSV file")

# Sidebar settings
st.sidebar.header("Model Hyperparameters")

max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

# Dataset selection
data_option = st.radio(
    "Choose Dataset:",
    ["Iris Dataset (Default)", "Upload CSV"]
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
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    st.subheader("Iris Dataset Preview")
    st.dataframe(df.head())

    X = df.drop(columns=["target"])
    y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
st.subheader("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write(f" **Accuracy:** {accuracy:.2f}")

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Decision Tree Visualization
st.subheader(" Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(16, 9))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(c) for c in np.unique(y)],
    filled=True,
    rounded=True,
    ax=ax
)

st.pyplot(fig)

st.success("Decision Tree Classifier trained and visualized successfully")
