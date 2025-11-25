from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    RandomForestClassifier,
    AdaBoostClassifier,
)


# Random Forest
def get_random_forest(task_type="regression"):
    # classification
    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
        )

    # regression
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )


# AdaBoost
def get_adaboost(task_type="regression"):
    # classification
    if task_type == "classification":
        return AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.8,
            random_state=42,
        )

    # regression
    return AdaBoostRegressor(
        n_estimators=200,
        learning_rate=0.8,
        random_state=42,
    )
