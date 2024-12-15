import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load the data
data_path = './data/2.elevator_failure_prediction.xlsx'
data = pd.read_excel(data_path, sheet_name='data')

# Create results folder if it doesn't exist 
results_folder = './results'
os.makedirs(results_folder, exist_ok=True)

# Data summary
summary = data.describe()
data_head = data.head()
missing_values = data.isnull().sum()

# Save data summary and missing values to CSV
summary.to_csv(f"{results_folder}/data_summary.csv")
data_head.to_csv(f"{results_folder}/data_head.csv", index=False)
missing_values.to_csv(f"{results_folder}/missing_values.csv", header=["Missing Count"])

# Handle missing values
data['Temperature'] = data['Temperature'].fillna(data['Temperature'].median())
data['Sensor2'] = data['Sensor2'].fillna(data['Sensor2'].median())

# Drop the 'Time' column since it's not a feature for prediction
data = data.drop(columns=['Time'])

# Separate features (X) and target (y)
X = data.drop(columns=['Status'])
y = data['Status']

# Check class distribution
class_distribution_before = y.value_counts()
class_distribution_before.to_csv(f"{results_folder}/class_distribution_before.csv", header=["Count"])

# Address class imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Check class distribution after resampling
class_distribution_after = pd.Series(y_resampled).value_counts()
class_distribution_after.to_csv(f"{results_folder}/class_distribution_after.csv", header=["Count"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Classification report with `zero_division` set to 0
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

# Save evaluation metrics
metrics = pd.DataFrame({"Metric": ["Accuracy", "ROC AUC Score"], "Value": [accuracy, roc_auc]})
metrics.to_csv(f"{results_folder}/evaluation_metrics.csv", index=False)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{results_folder}/classification_report.csv")

# Plot ROC curve (for each class)
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba.shape[1]):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} ROC Curve")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{results_folder}/roc_curve.png")
plt.close()

# Feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Save feature importances to CSV
feature_importance.to_csv(f"{results_folder}/feature_importance.csv", index=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importances")
plt.savefig(f"{results_folder}/feature_importance.png")
plt.close()

print("Results saved to 'results' folder.")
