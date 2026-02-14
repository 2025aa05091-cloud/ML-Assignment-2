# Mobile Price Classification App

### a. Problem Statement
The objective of this assignment is to develop an end-to-end machine learning workflow to classify mobile phone price ranges based on hardware specifications. This involves implementing six distinct classification models, evaluating them across multiple performance metrics, and deploying the solution via a Streamlit web application.

### b. Dataset Description [1 Mark]
* **Dataset**: Mobile Price Classification (Public Repository).
* **Instances**: 2,000 (Exceeds minimum requirement of 500).
* **Features**: 20 (Exceeds minimum requirement of 12).
* **Target Variable**: `price_range` (Values 0-3 representing cost categories).

### c. Models Used and Comparison Table [6 Marks]
The following models were implemented using the same dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9750 | 0.9996 | 0.9759 | 0.9750 | 0.9750 | 0.9669 |
| Decision Tree | 0.8325 | 0.8858 | 0.8337 | 0.8325 | 0.8316 | 0.7769 |
| kNN | 0.5300 | 0.7629 | 0.5698 | 0.5300 | 0.5407 | 0.3789 |
| Naive Bayes | 0.7975 | 0.9560 | 0.8061 | 0.7975 | 0.7994 | 0.7313 |
| Random Forest (Ensemble) | 0.8925 | 0.9826 | 0.8961 | 0.8925 | 0.8933 | 0.8572 |
| XGBoost (Ensemble) | 0.9050 | 0.9913 | 0.9062 | 0.9050 | 0.9050 | 0.8735 |

### d. Performance Observations [3 Marks]
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest accuracy, indicating that price ranges are linearly separable in this feature space. |
| **Decision Tree** | Captured non-linear patterns but showed significantly lower accuracy than ensemble methods. |
| **kNN** | Weakest performance; likely due to high dimensionality (20 features) affecting distance metrics. |
| **Naive Bayes** | Solid baseline performance; efficiently handled the multi-class nature of the target. |
| **Random Forest** | High stability and robustness; bagging helped reduce error compared to a single Decision Tree. |
| **XGBoost** | Second best performer; gradient boosting effectively optimized complex interactions between hardware specs. |

---

## Project Structure
- `app.py`: Interactive Streamlit application.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `2025AA05091-ml-assignment2.ipynb`: Source code for model training.
- `test_data.csv`: Sample data for testing the Streamlit app.
- `model/`: Directory containing trained model files (.pkl).

## Setup and Usage
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt