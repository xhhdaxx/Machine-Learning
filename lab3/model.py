from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(criterion="gini", max_depth=None, random_state=42):
    """
    构建决策树模型
    criterion: 划分标准 ['gini', 'entropy', 'log_loss']
    max_depth: 树深度限制
    """
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )
