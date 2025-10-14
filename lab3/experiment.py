# experiment.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from data_loader import load_dataset
from model import build_decision_tree
from evaluate import evaluate_model


def run_experiment(datasets, criteria, depths):
    """在多个数据集上运行不同参数组合的决策树实验"""
    results = []

    for name in datasets:
        data, X_train, X_test, y_train, y_test = load_dataset(name)
        print(f"\n数据集: {name.upper()}")
        print("=" * 40)

        for criterion in criteria:
            for depth in depths:
                model = build_decision_tree(criterion=criterion, max_depth=depth)
                model.fit(X_train, y_train)
                acc = evaluate_model(model, X_test, y_test)
                results.append((name, criterion, depth, acc))
                print(f"criterion={criterion:8s}, max_depth={str(depth):5s} → acc={acc:.4f}")

    return results


def visualize_tree(dataset_name="iris", criterion="entropy", max_depth=4):
    """绘制决策树结构可视化"""
    data, X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    model = build_decision_tree(criterion=criterion, max_depth=max_depth)
    model.fit(X_train, y_train)

    plt.figure(figsize=(16, 8))
    plot_tree(
        model,
        filled=True,
        feature_names=data.feature_names,
        class_names=data.target_names,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree Visualization ({dataset_name}, {criterion})")
    plt.show()


def visualize_decision_boundary(dataset_name="iris", feature_idx=[0, 2], criterion="gini", max_depth=4):
    """绘制二维特征下的决策边界图"""
    data, _, _, _, _ = load_dataset(dataset_name)
    X = data.data[:, feature_idx]
    y = data.target

    clf = build_decision_tree(criterion=criterion, max_depth=max_depth)
    clf.fit(X, y)

    # 生成网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    # 预测每个网格点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.rainbow)
    plt.title(f"Decision Boundary ({dataset_name}, criterion={criterion}, depth={max_depth})")
    plt.xlabel(data.feature_names[feature_idx[0]])
    plt.ylabel(data.feature_names[feature_idx[1]])
    plt.show()
