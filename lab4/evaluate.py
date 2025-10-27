from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(model, X_test, y_test):
    # 获取模型预测结果
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 打印更详细的评估指标：精确率、召回率、F1分数
    report = classification_report(y_test, y_pred)

    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    return accuracy, report, cm
