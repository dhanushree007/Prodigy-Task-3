# Prodigy-Task-3

## Overview

This project uses a decision tree classifier to predict whether a bank customer will subscribe to a term deposit (binary classification). The data is preprocessed, split into training and testing sets, and a decision tree model is trained and evaluated. Key metrics like accuracy, classification report, and feature importance are analyzed, and visualizations are provided for the decision tree and box plots of relevant features.

---

## Requirements

To run this notebook, the following libraries must be installed:

- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning algorithms.
- `matplotlib`: For plotting and visualizations.
- `seaborn`: For statistical data visualization.

You can install all required libraries using the command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Dataset

The dataset used for this analysis is the **Bank Marketing Dataset**. It includes information about customers and the outcome of whether they subscribed to a term deposit or not. The dataset can be loaded via a CSV file named `bank.csv`. 

---

## Key Features

- **age**: Age of the customer.
- **job**: Type of job the customer has.
- **marital**: Marital status of the customer.
- **education**: Education level.
- **default**: Does the customer have credit in default? (binary)
- **balance**: Account balance.
- **housing**: Does the customer have a housing loan? (binary)
- **loan**: Does the customer have a personal loan? (binary)
- **contact**: Contact communication type.
- **day**: Last contact day of the month.
- **month**: Last contact month of the year.
- **duration**: Last contact duration in seconds.
- **campaign**: Number of contacts performed during this campaign.
- **pdays**: Number of days since the customer was last contacted.
- **previous**: Number of contacts before this campaign.
- **poutcome**: Outcome of the previous marketing campaign.
- **deposit**: Has the customer subscribed to a term deposit? (binary: "yes", "no").

---

## Steps in the Analysis

### 1. **Loading Data**
   - The data is read using `pandas`, and an initial exploration is performed to understand the data shape and types of columns.
   - Missing value checks and data types of columns are printed.

### 2. **Data Preprocessing**
   - A `LabelEncoder` is used to convert categorical variables into numerical values.
   - A box plot is created for the **job** and **age** columns for quick visualization.
   - The data is cleaned and prepared for model training by converting categorical features.

### 3. **Train-Test Split**
   - The dataset is split into features (`X`) and target (`y`) variables, where the target is whether the customer subscribed to the term deposit (`deposit`).
   - The data is split into a training set (80%) and a test set (20%) using `train_test_split`.

### 4. **Model Training**
   - A `DecisionTreeClassifier` from `scikit-learn` is used to train the model.
   - The model is trained on the training data.

### 5. **Model Evaluation**
   - Predictions are made on the test set.
   - The accuracy of the model is calculated.
   - A classification report and confusion matrix are printed to evaluate model performance.

### 6. **Visualization**
   - A decision tree is visualized with a depth limit of 3 for better clarity.
   - Feature importance is extracted from the trained model and displayed in descending order.

---

## Key Outputs

1. **Accuracy Score**: The overall accuracy of the model in predicting whether customers will subscribe to the term deposit.
2. **Classification Report**: Precision, recall, F1-score, and support for each class.
3. **Confusion Matrix**: A matrix showing true positives, false positives, true negatives, and false negatives.
4. **Decision Tree Visualization**: A plot of the decision tree with a limited depth of 3.
5. **Feature Importance**: A ranked list of the most important features in predicting whether a customer will subscribe.

---

## How to Run

1. Clone or download the repository.
2. Load the dataset (`bank.csv`) into the same directory as the notebook.
3. Open the notebook in Jupyter or Google Colab.
4. Install the necessary libraries.
5. Run each code cell sequentially to train the decision tree model and view the results.

---

## Contact

For any questions or issues regarding this project, feel free to reach out at dhanushree2607@gmail.com.
