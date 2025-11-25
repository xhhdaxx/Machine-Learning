import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data(file_path="Admission_Predict_Ver1.1.csv", normalize=True, pca_dim=None, test_size=0.2, random_state=42):
    """
    加载数据集 Admission_Predict_Ver1.1.csv，支持归一化与PCA降维
    """
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Serial No."])  # 去掉序号列

    X = df.drop(columns=["Chance of Admit"])
    y = df["Chance of Admit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 是否归一化
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 是否PCA降维
    if pca_dim:
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test
