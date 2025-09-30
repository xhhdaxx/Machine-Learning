from .data_loader import load_digit_dataset, load_dating_dataset
from .model import build_knn_model
from .evaluate import evaluate_model, plot_pr_curve, plot_roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def run_experiment(dataset_type, train_path, test_path=None, k=3, p=2,
                   plot_type=None, return_metrics=False, return_plot=True, train_ratio=0.8):
    """
    Args:
        dataset_type: 'digits' 或 'dating'
        train_path: train dataset
        test_path: test dataset
    """
    print(f"Running {dataset_type} experiment with k={k}, p={p}...")

    # 1. 加载数据
    if dataset_type == "digits":
        X_train, y_train = load_digit_dataset(train_path)
        X_test, y_test = load_digit_dataset(test_path)
    elif dataset_type == "dating":
        X_train, y_train = load_dating_dataset(train_path)
        # 简单划分 80% 训练 + 20% 测试
        split = int(train_ratio * len(X_train))
        X_test, y_test = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]
    else:
        raise ValueError("Unknown dataset type!")

    # 2. 建立模型
    model = build_knn_model(n_neighbors=k, p=p)
    model.fit(X_train, y_train)

    # 3. 评估
    evaluate_model(model, X_test, y_test)

    # 4. 绘制曲线
    if return_plot:
        if plot_type == 'pr':
            plot_pr_curve(model, X_test, y_test)
        elif plot_type == 'roc':
            plot_roc_curve(model, X_test, y_test)

    # 5. 是否返回结果
    if return_metrics:
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        # 提取需要的指标
        metrics_dict = {
            "accuracy": report["accuracy"],
            "macro avg": report["macro avg"],
            "weighted avg": report["weighted avg"]
        }
        return metrics_dict


def param_sweep_experiment(dataset_name, train_path, test_path,
                           param_name="k", param_values=None,
                           fixed_k=3, fixed_p=2, plot_type="pr"):
    """
    Args:
        dataset_name: digits or dating
        train_path: train dataset
        test_path: test dataset
        param_name: "k" or "p"
        param_values: params list
        fixed_k: if sweep p -> k value
        fixed_p: if sweep k -> p value
        plot_type: "pr" or "roc"
    """
    if param_values is None:
        param_values = list(range(1, 11))  # 默认 sweep 1~10

    acc_list, macro_list, weighted_list = [], [], []

    for val in param_values:
        if param_name == "k":
            metrics = run_experiment(dataset_name, train_path, test_path,
                                     k=val, p=fixed_p, plot_type=plot_type, return_metrics=True, return_plot=False)
        elif param_name == "p":
            metrics = run_experiment(dataset_name, train_path, test_path,
                                     k=fixed_k, p=val, plot_type=plot_type, return_metrics=True, return_plot=False)
        else:
            raise ValueError("param_name must be 'k' or 'p'")

        acc_list.append(metrics["accuracy"])
        macro_list.append(metrics["macro avg"]["f1-score"])
        weighted_list.append(metrics["weighted avg"]["f1-score"])

    # 画趋势图
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, acc_list, marker="o", label="Accuracy")
    plt.plot(param_values, macro_list, marker="s", label="Macro avg F1")
    plt.plot(param_values, weighted_list, marker="^", label="Weighted avg F1")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    if param_name == "k":
        plt.title(f"KNN: effect of {param_name} on {dataset_name} (p={fixed_p})")
    if param_name == "p":
        plt.title(f"KNN: effect of {param_name} on {dataset_name} (k={fixed_k})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()


def train_ratio_experiment(train_path, ratios=[0.5, 0.6, 0.7, 0.8, 0.9],
                           k=3, p=2, plot_type=None):
    """
    Args:
        train_path: dating dataset
        ratios: different train ratios
        plot_type: 'pr' or 'roc'
    """
    accuracies, macro_f1s, weighted_f1s = [], [], []

    for r in ratios:
        metrics = run_experiment("dating", train_path, k=k, p=p,
                                 plot_type=plot_type,
                                 return_metrics=True, return_plot=False,
                                 train_ratio=r)
        accuracies.append(metrics["accuracy"])
        macro_f1s.append(metrics["macro avg"]["f1-score"])
        weighted_f1s.append(metrics["weighted avg"]["f1-score"])

    # 绘制趋势图
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, accuracies, marker='o', label="Accuracy")
    plt.plot(ratios, macro_f1s, marker='s', label="Macro F1")
    plt.plot(ratios, weighted_f1s, marker='^', label="Weighted F1")

    plt.xlabel("Train Ratio")
    plt.ylabel("Score")
    plt.title("Effect of Train/Test Split on Dating Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()
