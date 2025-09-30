from src.experiment import run_experiment, param_sweep_experiment, train_ratio_experiment


def main():
    # 1. 手写数字识别实验
    run_experiment("digits", "./data/trainingDigits", "./data/testDigits",
                   k=2, p=2, plot_type="pr")

    # 2. 约会数据集实验
    # run_experiment("dating", "./data/datingTestSet.txt",
    #                k=3, p=2, plot_type="roc", train_ratio=0.8)

    # 3. 探究不同训练集比例对实验结果的影响
    # train_ratio_experiment("./data/datingTestSet.txt",
    #                        ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                        k=5, p=2)

    # 4. 探究不同k值对实验结果的影响
    # param_sweep_experiment("digits", "./data/trainingDigits", "./data/testDigits",
    #                        param_name="k", param_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                        fixed_p=2, plot_type="pr")

    # 5. 探究不同p值对实验结果的影响
    # param_sweep_experiment("digits", "./data/trainingDigits", "./data/testDigits",
    #                        param_name="p", param_values=[1, 2],
    #                        fixed_k=9, plot_type="pr")


if __name__ == "__main__":
    main()
