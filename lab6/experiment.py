from data_loader import load_admission_data, load_pulsar_data
from model import get_random_forest, get_adaboost
from evaluate import evaluate_model


def run_experiment(csv_path, name):
    # 1. 选择数据集
    if name == "1.Admission_Predict_Ver1.1.csv":
        X_train, X_test, y_train, y_test = load_admission_data(csv_path)
        task_type = "regression"
    elif name == "2.pulsar_stars.csv":
        X_train, X_test, y_train, y_test = load_pulsar_data(csv_path)
        task_type = "classification"
    else:
        raise ValueError("Unsupported dataset name")

    # 2. 初始化模型
    rf = get_random_forest(task_type)
    ada = get_adaboost(task_type)

    # 3. 训练模型
    rf.fit(X_train, y_train)
    ada.fit(X_train, y_train)

    # 4. 评估模型
    rf_result = evaluate_model(rf, X_test, y_test, task_type)
    ada_result = evaluate_model(ada, X_test, y_test, task_type)

    return rf_result, ada_result


def run_trials(dataset):
    rf_res, ada_res = run_experiment(dataset["path"], dataset["name"])
    return {
        "Random Forest": rf_res,
        "AdaBoost": ada_res,
    }


def print_results(title, metrics, results):
    print(f"======= {title} =======")
    if title == "Admission 回归任务":
        print("Model\t\t\tMAE\t\t\tMSE\t\t\tR2")
    elif title == "Pulsar 分类任务":
        print("Model\t\t\tAccuracy\tPrecision\tRecall\t\tF1")

    # Random Forest
    rf_row = ["Random Forest"] + [f"{results['Random Forest'][m]:.5f}\t" for m in metrics]
    print("\t".join(rf_row))
    # AdaBoost
    ada_row = ["AdaBoost\t"] + [f"{results['AdaBoost'][m]:.5f}\t" for m in metrics]
    print("\t".join(ada_row))

    print()


def contrast_experiment(DATASETS):
    for dataset in DATASETS:
        dataset_results = run_trials(dataset)
        print_results(dataset["title"], dataset["metrics"], dataset_results)
