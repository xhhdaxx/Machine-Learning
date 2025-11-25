from experiment import contrast_experiment

DATASETS = [
    {
        "name": "1.Admission_Predict_Ver1.1.csv",
        "path": "/Users/hexinhao/data/ml_dataset/1.Admission_Predict_Ver1.1.csv",
        "metrics": ["MAE", "MSE", "R2"],
        "title": "Admission 回归任务",
    },
    {
        "name": "2.pulsar_stars.csv",
        "path": "/Users/hexinhao/data/ml_dataset/2.pulsar_stars.csv",
        "metrics": ["Accuracy", "Precision", "Recall", "F1"],
        "title": "Pulsar 分类任务",
    },
]

if __name__ == "__main__":
    contrast_experiment(DATASETS)
