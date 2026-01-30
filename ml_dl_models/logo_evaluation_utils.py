import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_logo(model_builder, X, y, groups, scaler=None):
    """
    model_builder: function that returns a fresh untrained model
    scaler: None for RF, StandardScaler instance for MLP
    """

    unique_students = np.unique(groups)

    metrics = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "roc_auc": []
    }

    confusion_matrices = {}

    for student in unique_students:
        test_mask = groups == student
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if scaler is not None:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_builder()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_test).ravel()
        )

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))

        confusion_matrices[student] = confusion_matrix(y_test, y_pred)

    return metrics, confusion_matrices
