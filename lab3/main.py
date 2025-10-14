# main.py
from experiment import run_experiment, visualize_tree, visualize_decision_boundary

if __name__ == "__main__":
    datasets = ["iris", "wine"]
    criteria = ["gini", "entropy", "log_loss"]
    depths = [2, 4, 6, None]

    results = run_experiment(datasets, criteria, depths)

    print("\n实验结果汇总：")
    for r in results:
        print(f"Dataset={r[0]:5s} | criterion={r[1]:8s} | depth={str(r[2]):5s} | acc={r[3]:.4f}")

    dataset = "iris"
    criterion = "entropy"
    max_depth = 4

    # 决策树结构可视化
    visualize_tree(dataset_name=dataset, criterion=criterion, max_depth=max_depth)

    # 决策边界可视化（二维特征）
    visualize_decision_boundary(dataset_name=dataset, feature_idx=[0, 2], criterion=criterion, max_depth=max_depth)
