from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def get_model(model_name="svm"):
    """
    根据模型名称返回模型实例
    """
    if model_name == "linear":
        return LinearRegression()
    elif model_name == "svm":
        return SVR(kernel='rbf', C=10, gamma='scale')
    elif model_name == "tree":
        return DecisionTreeRegressor(max_depth=5, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
