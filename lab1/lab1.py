# coding:utf-8
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# 1.加载数据集
data = load_iris()
datas = data.data
labels = data.target

# 2.初始化列表以存储多次运行的结果
test_size = 0.3  # 30%作为测试集
n_runs = 30  # 运行30次
test_accuracies = []
biases = []
variances = []

# 3. 查看数据集信息
print("=== 数据集分析 ===")
print("数据集大小:", datas.shape[0])
print("测试集比例:", test_size * 100, "%")
print("数据集特征:", datas.shape[1])
print("特征名称:", data.feature_names)
print("标签数量:", len(np.unique(labels)))
print("标签名称:", data.target_names)
print("前5行数据:\n", data.data[:5])

# 4.进行30次训练
print("=== 方法：训练集/测试集划分 ===")
print("Time\t Test Accuracy")
for i in range(n_runs):
    # 使用不同的随机种子划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        datas, labels, test_size=test_size, random_state=i, stratify=labels
    )

    # 5.选择分类模型 - 逻辑回归
    model = LogisticRegression(max_iter=200)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算测试集精度
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_accuracy)

    # 6.计算偏差和方差
    # 使用交叉验证来估计模型性能
    cv_scores = cross_val_score(model, datas, labels, cv=KFold(n_splits=5, shuffle=True, random_state=i),
                                scoring='accuracy')

    # 偏差 (bias) = 1 - 平均交叉验证精度
    bias = 1 - np.mean(cv_scores)
    biases.append(bias)

    # 方差 (variance) = 交叉验证精度的标准差
    variance = np.std(cv_scores)
    variances.append(variance)

    print(f"{i + 1}\t\t {test_accuracy:.5f}\t")

# 7.输出结果
print("=== 实验结果 ===")
print("平均测试集精度: {:.5f}".format(np.mean(test_accuracies)))
print("测试集精度标准差: {:.5f}".format(np.std(test_accuracies)))
print("-----------------------")
print("平均偏差: {:.5f}".format(np.mean(biases)))
print("偏差标准差: {:.5f}".format(np.std(biases)))
print("-----------------------")
print("平均方差: {:.5f}".format(np.mean(variances)))
print("方差标准差: {:.5f}".format(np.std(variances)))
