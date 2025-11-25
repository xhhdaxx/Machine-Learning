from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test, task_type="regression"):
    """
    评估模型性能:
        - 回归: MAE、MSE、R2
        - 分类: Accuracy、Precision、Recall、F1
    """
    pred = model.predict(X_test)

    # classification
    if task_type == "classification":
        return {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        }

    # regression
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }
