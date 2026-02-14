import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Mobile Price Classification - ML Assignment 2")

# Sidebar for data upload
st.sidebar.header("Upload Data")
st.sidebar.markdown("### Download Test Data")

# Provide download link for users to test the implementation
try:
    with open("test_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download test_data.csv",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.sidebar.error("test_data.csv not found in root directory.")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# Model selection dropdown
model_option = st.selectbox(
    "Select Model for Evaluation",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
)

# Mapping selection to saved .pkl files in the model folder
model_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

if uploaded_file is not None:
    # Load user data
    data = pd.read_csv(uploaded_file)
    X_test = data.drop('price_range', axis=1)
    y_test = data['price_range']
    
    # Load selected model and the saved scaler
    model = joblib.load(model_map[model_option])
    scaler = joblib.load("model/scaler.pkl")
    
    # --- PREDICTION LOGIC ---
    # Apply scaling only for distance-based/linear models as per notebook training
    if model_option in ["Logistic Regression", "kNN"]:
        X_test_final = scaler.transform(X_test)
    else:
        X_test_final = X_test
        
    y_pred = model.predict(X_test_final)
    # ------------------------

    # Display evaluation metrics
    st.subheader(f"Metrics for {model_option}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
    
    col4, col5 = st.columns(2)
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
    col5.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
    
    # Confusion matrix and classification report
    st.markdown("---")
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
else:
    st.info("Please upload the test_data.csv file to proceed.")