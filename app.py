import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Split into features (X) and target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split into training/testing, 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict the likelihood of diabetes.")

# Collect user input
input_data = []
for col in X.columns:
    value = st.number_input(f"Enter {col}", min_value=0.0, step=0.1)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    prediction = model.predict([input_data])
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: {result}")

# Model Performance
st.header("Model Performance")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
