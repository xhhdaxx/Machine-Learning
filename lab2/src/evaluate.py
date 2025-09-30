from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, X_test, y_test):
    """ 输出分类结果报告 """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def plot_pr_curve(model, X_test, y_test):
    """ 绘制PR曲线（针对多分类做one-vs-rest处理） """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import PrecisionRecallDisplay

    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.title(f"PR Curve (class {classes[i]})")
        plt.show()


def plot_roc_curve(model, X_test, y_test):
    """ 绘制ROC曲线（针对多分类做one-vs-rest处理） """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import RocCurveDisplay

    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        disp.plot()
        plt.title(f"ROC Curve (class {classes[i]})")
        plt.show()
