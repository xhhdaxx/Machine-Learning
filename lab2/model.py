from sklearn.neighbors import KNeighborsClassifier


def build_knn_model(n_neighbors=3, p=2):
    """
    params:
        n_neighbors: K value
        p: (1=L1, 2=L2)
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
