import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="Bank_Personal_Loan_preprocessing.csv")
args = parser.parse_args()
data_path = args.data_path

df = pd.read_csv(data_path)
X = df.drop(columns=["Personal Loan"])
y = df["Personal Loan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('Bank_Personal_Loan_Experiment')
mlflow.set_tracking_uri('http://127.0.0.1:5000')

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
