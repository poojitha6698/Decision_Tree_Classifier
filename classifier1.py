import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Page config
st.set_page_config(
    page_title="California Housing Classifier",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("California Housing Price Classifier")
st.write("Decision Tree Classification using Streamlit")

st.divider()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("california_housing_test.csv")

df = load_data()

# Create classification target
median_value = df["median_house_value"].median()
df["price_class"] = (df["median_house_value"] >= median_value).astype(int)

# Features and target
X = df.drop(columns=["median_house_value", "price_class"])
y = df["price_class"]

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

criterion = st.sidebar.selectbox(
    "Split Criterion",
    ("gini", "entropy")
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train classifier
model = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display metrics
st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy:.3f}")

st.write("**Confusion Matrix**")
st.dataframe(
    pd.DataFrame(
        cm,
        columns=["Predicted Low", "Predicted High"],
        index=["Actual Low", "Actual High"]
    )
)

st.divider()

# User input section
st.subheader("Classify House Price")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(
        col,
        value=float(df[col].mean())
    )

input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Classify"):
    prediction = model.predict(input_df)[0]
    label = "High Value House" if prediction == 1 else "Low Value House"
    st.success(f"Prediction: **{label}**")

st.divider()

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(df.head())
