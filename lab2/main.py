from src.experiment import run_experiment, param_sweep_experiment


def main():
    # 1. 手写数字识别实验
    # run_experiment("digits", "./data/trainingDigits", "./data/testDigits",
    #                k=10, p=2, plot_type="pr")

    # 2. 约会数据集实验
    run_experiment("dating", "./data/datingTestSet.txt",
                   k=10, p=2, plot_type="roc", train_ratio=0.1)

    # 3. 探究不同k值对实验结果的影响
    # param_sweep_experiment("digits", "./data/trainingDigits", "./data/testDigits",
    #                        param_name="k", param_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                        fixed_p=2, plot_type="pr")

    # 4. 探究不同p值对实验结果的影响
    # param_sweep_experiment("digits", "./data/trainingDigits", "./data/testDigits",
    #                        param_name="p", param_values=[1, 2],
    #                        fixed_k=9, plot_type="pr")


if __name__ == "__main__":
    main()
