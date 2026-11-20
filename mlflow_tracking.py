import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data=pd.read_csv("dataset/students_clean.csv")

x = data[["study_hours", "attendance", "previous_marks"]]
y = data["result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
mlflow.set_experiment("Student_performance")

with mlflow.start_run():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

mlflow.log_param("model_name","RandomForestClassifier")
mlflow.log_param("n_estimator",100)
mlflow.log_param("random_state",42)

