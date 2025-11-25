from data_loader import load_data
from model import get_model
from evaluate import evaluate_model


def run_experiment(file_path="Admission_Predict_Ver1.1.csv", normalize=True, pca_dim=None, model_name="svm"):
    """
    运行一次实验（加载数据 -> 训练模型 -> 评估性能）
    """
    X_train, X_test, y_train, y_test = load_data(file_path, normalize, pca_dim)
    model = get_model(model_name)
    model.fit(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    print(f"Model: {model_name.upper()}, Normalize: {normalize}, PCA: {pca_dim}")
    print(f" → MAE: {results['MAE']:.4f}, MSE: {results['MSE']:.4f}, RMSE: {results['RMSE']:.4f}, R2: {results['R2']:.4f}")
    print("-" * 60)
    return results
