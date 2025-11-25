from experiment import run_experiment

if __name__ == "__main__":
    file_path = "/Users/hexinhao/data/admission/Admission_Predict_Ver1.1.csv"

    configs = [
        # 不归一化 + 无PCA降维
        {"normalize": False, "pca_dim": None, "model": "svm"},
        {"normalize": False, "pca_dim": None, "model": "linear"},
        {"normalize": False, "pca_dim": None, "model": "tree"},

        # 归一化 + 无PCA降维
        {"normalize": True, "pca_dim": None, "model": "svm"},
        {"normalize": True, "pca_dim": None, "model": "linear"},
        {"normalize": True, "pca_dim": None, "model": "tree"},

        # 不归一化 + PCA降维
        {"normalize": False, "pca_dim": 5, "model": "svm"},
        {"normalize": False, "pca_dim": 5, "model": "linear"},
        {"normalize": False, "pca_dim": 5, "model": "tree"},

        # 归一化 + PCA降维
        {"normalize": True, "pca_dim": 5, "model": "svm"},
        {"normalize": True, "pca_dim": 5, "model": "linear"},
        {"normalize": True, "pca_dim": 5, "model": "tree"},
    ]

    print("==== Admission Prediction Experiment ====\n")

    # 1. 单次实验
    # run_experiment(
    #     file_path=file_path,
    #     normalize=True,
    #     pca_dim=None,
    #     model_name="svm"
    # )

    # 2. 对比实验
    for cfg in configs:
        run_experiment(
            file_path=file_path,
            normalize=cfg["normalize"],
            pca_dim=cfg["pca_dim"],
            model_name=cfg["model"]
        )
