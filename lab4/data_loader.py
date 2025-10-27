import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(data_dir):
    ham_dir = os.path.join(data_dir, 'ham')
    spam_dir = os.path.join(data_dir, 'spam')

    # 读取ham和spam文件夹中的所有文本
    ham_files = [os.path.join(ham_dir, f) for f in os.listdir(ham_dir)]
    spam_files = [os.path.join(spam_dir, f) for f in os.listdir(spam_dir)]

    # 读取每个文件的内容，指定编码为 'ISO-8859-1'
    ham_texts = [open(f, encoding='ISO-8859-1').read() for f in ham_files]
    spam_texts = [open(f, encoding='ISO-8859-1').read() for f in spam_files]

    # 标签
    labels = [0] * len(ham_texts) + [1] * len(spam_texts)  # 0表示ham, 1表示spam

    texts = ham_texts + spam_texts
    return texts, labels


def prepare_data(data_dir):
    texts, labels = load_data(data_dir)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
