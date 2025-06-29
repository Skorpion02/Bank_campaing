# Bank_campaing

![GitHub Repo stars](https://img.shields.io/github/stars/Skorpion02/Bank_campaing?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/Skorpion02/Bank_campaing?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Skorpion02/Bank_campaing?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/Skorpion02/Bank_campaing?style=flat-square)
![License](https://img.shields.io/github/license/Skorpion02/Bank_campaing?style=flat-square)

---

## 🚀 Project Overview

**Bank_campaing** is an open-source solution for managing, analyzing, and optimizing marketing campaigns in banking and financial institutions.  
The project provides tools to efficiently plan, execute, and evaluate the impact of marketing initiatives, helping banks reach their target audience and maximize return on investment (ROI).

---

## 🎯 Main Objectives

- **Facilitate Campaign Management:**  
  Centralize campaign creation, scheduling, and tracking in one platform.

- **Enhance Customer Segmentation:**  
  Use data-driven tools to identify, segment, and target potential clients for each campaign.

- **Analyze Results in Real Time:**  
  Access dashboards and detailed reports to monitor campaign effectiveness, conversion rates, and customer feedback.

- **Automate Repetitive Tasks:**  
  Schedule communications, automate follow-ups, and set up triggers based on client actions.

- **Secure and Compliant:**  
  Ensure that all data is managed securely and meets banking compliance standards.

---

## ✨ Key Features

- 📊 Campaign tracking and analytics
- 👥 Customer segmentation and targeting
- 🗓️ Scheduling and automation of campaigns
- 📈 Real-time performance dashboards
- 🔒 Secure data management and audit logs
- 📨 Multi-channel communication (email, SMS, notifications)
- 🧩 Integration capabilities with CRM and banking systems
- 📝 Customizable templates for different campaign types

---

## 🧩 Example Use Cases

- Launching seasonal promotions for credit cards or loans
- Sending personalized offers to high-value clients
- Monitoring the success of digital vs. traditional marketing efforts
- Automating reminders for expiring products or loyalty programs

---

## 📒 Notebook Contents

The `bank_campaing.ipynb` notebook is structured into the following main sections:

### 1. Working Environment Setup
* **Library Imports:** Essential libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib.pyplot`, `seaborn`), and machine learning (`sklearn`, `lightgbm`) are imported. This includes modules for preprocessing (`StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`, `ColumnTransformer`), model selection (`train_test_split`, `GridSearchCV`), metrics (`roc_auc_score`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`), and classification algorithms (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `LGBMClassifier`).

### 2. Data Loading
* **Dataset Loading:** Both the training (`train.csv`) and validation (`test.csv`) datasets are loaded directly from Google Drive using `gdown`. This ensures that the notebook can be easily run in environments like Google Colab.

### 3. Exploratory Data Analysis (EDA) and Preprocessing
* **Initial Inspection:** An initial review of the dataframe's information (`.info()`) is performed to identify data types and non-null values, as well as a statistical summary (`.describe()`).
* **Handling Missing Values:** Missing values in the dataset are identified and managed (although in this case, there appear to be no significant nulls in the main columns).
* **Variable Identification:** Columns are classified into numerical, categorical, and ordinal variables for appropriate preprocessing.
* **Feature Engineering:** New variables may be created from existing ones, or current ones transformed to improve model performance.
* **Categorical Variable Encoding:**
    * **Ordinal Encoding:** Applied to variables with an intrinsic order (e.g., 'education').
    * **One-Hot Encoding:** Used for nominal categorical variables without a specific order.
* **Numerical Feature Scaling:** `StandardScaler` is used to normalize numerical variables, which is crucial for the good performance of some machine learning algorithms.
* **Preprocessing Pipeline:** A `ColumnTransformer` is built to apply different transformations to different types of columns efficiently.

### 4. Model Training and Evaluation
* **Data Splitting:** The training data is split into training and testing sets for internal model validation.
* **Model Definition:** Several classification models are initialized, including:
    * `LogisticRegression`
    * `RandomForestClassifier`
    * `SVC` (Support Vector Classifier)
    * `LGBMClassifier` (Light Gradient Boosting Machine)
* **Parameter Definition for `GridSearchCV`:** Hyperparameter ranges are defined for each model to perform an exhaustive search.
* **Training and Optimization (Grid Search):** `GridSearchCV` is used to find the best hyperparameters for each model, evaluating performance with the AUC-ROC metric.
* **Best Model Selection:** The performance of the trained models is compared, and the one achieving the best AUC-ROC score is selected.
* **Detailed Evaluation:** The best model is evaluated on the test set using metrics such as `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `roc_auc_score`.

### 5. Prediction and Submission File Generation
* **Validation Data Transformation:** The validation dataset (`test.csv`) is preprocessed using the same `ColumnTransformer` fitted on the training data.
* **Predictions:** The best-trained model is used to predict the subscription probabilities (`output`) on the transformed validation dataset.
* **`submission.csv` Generation:** A CSV file (`submission.csv`) is created with the `id` and `output` (predicted probability) columns, ready for submission in competitions or for further use.

---

## 📊 Results and Conclusions

The primary goal of this project is to develop and train a robust classification model capable of accurately predicting customer subscription to a term deposit. The notebook provides a detailed exploration of different machine learning algorithms and leverages hyperparameter optimization techniques to achieve the best possible performance, primarily evaluated by the Area Under the Receiver Operating Characteristic curve (AUC-ROC). The generated submission.csv file serves as the tangible output, encapsulating the predicted probabilities for each customer in the validation set.

The project evaluates the performance of the selected model on the validation set using several metrics. Recently, an additional step was incorporated to explicitly compute and display the model’s accuracy. The following code illustrates how the accuracy is calculated and shown:

```python
y_true = y_val
probs_val = best_model.predict_proba(X_val)[:,1]
y_pred = (probs_val >= best_t).astype(int)

accuracy = accuracy_score(y_true, y_pred)
print(f"La precisión de tu modelo es: {accuracy * 100:.2f}%")
```

With this approach, the model achieved an accuracy of **86.95%** on the validation dataset. This result indicates a strong performance in classifying the bank campaign data, and provides a clear and direct metric for evaluating and comparing future models.

---

## 🛠️ Technologies Used

The `bank_campaing.ipynb` file uses the following languages, frameworks, and tools:

* ### Language: Python

* ### Frameworks/Libraries:
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning, including preprocessing, model selection, and various classification algorithms like Logistic Regression, Random Forest, and SVC)
* LightGBM (for gradient boosting classification)
* Matplotlib (for plotting)
* Seaborn (for statistical data visualization)
* Gdown (for downloading files from Google Drive)

* ### Tool: Jupyter Notebook (`.ipynb`)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions, bug fixes, or new features.

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## 📬 Contact

For questions or support, please contact [Skorpion02](https://github.com/Skorpion02).

---

⭐️ **If you found this project helpful, please give it a star!**

---

<div align="center">
  <b>Made with ❤️ by Skorpion02</b>
</div>
