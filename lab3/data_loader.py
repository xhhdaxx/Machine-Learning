from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


def load_dataset(name):
    """根据名称加载不同数据集"""
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError("Unsupported dataset. Choose from ['iris', 'wine']")

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    return data, X_train, X_test, y_train, y_test
