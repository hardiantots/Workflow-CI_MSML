import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="Bank_Personal_Loan_preprocessing.csv")
args = parser.parse_args()

def load_and_split_data(data, target_column, test_size=0.2, random_state=None):
    df = pd.read_csv(data)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train, X_test, y_train, y_test = load_and_split_data(args.data_path, 'Personal Loan')

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
report = classification_report(y_test, y_pred)

# Infer signature
input_example = X_test.iloc[[0]]
signature = infer_signature(X_test, y_pred)

# Log model and metrics
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    input_example=input_example,
    signature=signature
)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_text(report, "classification_report.txt")

# Save run ID
run_id = mlflow.active_run().info.run_id
print(f"MLflow Run ID for this execution: {run_id}")
with open("run_id.txt", "w") as f:
    f.write(run_id)
