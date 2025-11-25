import pandas as pd
from sklearn.model_selection import train_test_split


def load_admission_data(path):
    """
    读取 Admission 数据集并划分训练集与测试集
    """
    df = pd.read_csv(path)

    # 特征列（除 Chance of Admit）
    X = df.drop(columns=['Chance of Admit '])
    y = df['Chance of Admit ']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def load_pulsar_data(path):
    """
        读取 Pulsar 数据集并划分训练集与测试集
    """
    df = pd.read_csv(path)

    # 特征列（除 target_class）
    X = df.drop(columns=['target_class'])
    y = df['target_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
