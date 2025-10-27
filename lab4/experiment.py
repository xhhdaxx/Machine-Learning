from model import NaiveBayesClassifier, DecisionTreeClassifierWrapper
from data_loader import prepare_data
from evaluate import evaluate  # 导入evaluate函数


def run_experiment():
    data_dir = "/Users/hexinhao/data/email"
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(data_dir)

    # 贝叶斯分类器
    nb_model = NaiveBayesClassifier()
    nb_model.train(X_train, y_train)
    print("Evaluating Naive Bayes Classifier:")
    nb_accuracy, nb_report, nb_cm = evaluate(nb_model, X_test, y_test)

    # 决策树分类器
    dt_model = DecisionTreeClassifierWrapper()
    dt_model.train(X_train, y_train)
    print("\nEvaluating Decision Tree Classifier:")
    dt_accuracy, dt_report, dt_cm = evaluate(dt_model, X_test, y_test)

    # 结果返回或进一步分析
    return nb_accuracy, dt_accuracy
