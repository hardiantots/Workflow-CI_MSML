import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="Bank_Personal_Loan_preprocessing.csv")
args = parser.parse_args()

def load_and_split_data(data, target_column, test_size=0.2, random_state=None):
    df = pd.read_csv(data)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train, X_test, y_train, y_test = load_and_split_data(args.data_path, 'Personal Loan')

mlflow.set_experiment('Bank_Personal_Loan_Experiment')
mlflow.set_tracking_uri('http://127.0.0.1:5000')

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_text(report, 'classification_report.txt')
