import mlflow
import mlflow.sklearn
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


mlflow.set_tracking_uri(
    "file:///Users/dilshantharushika/Desktop/fake_news gnn")


df = pd.read_csv("creditcard_2023.csv")
X = df.drop("Class", axis=1)
y = df["Class"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


with mlflow.start_run(run_name="RF_CreditCardFraud"):

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 5)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Save and log the scaler
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")

    # Save and log the confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("conf_matrix.png")
    mlflow.log_artifact("conf_matrix.png")
