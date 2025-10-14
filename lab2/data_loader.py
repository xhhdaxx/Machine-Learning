import numpy as np
import os


def img2vector(filename):
    """ 将32x32的文本图像转化为1x1024的向量 """
    return_vec = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vec[0, 32 * i + j] = int(line_str[j])
    return return_vec


def load_digit_dataset(path):
    """ 加载数字数据集 """
    labels = []
    file_list = os.listdir(path)
    m = len(file_list)
    data_matrix = np.zeros((m, 1024))

    for i, filename in enumerate(file_list):
        # 文件名格式：{digit}_{index}.txt
        label = int(filename.split('_')[0])
        labels.append(label)
        data_matrix[i, :] = img2vector(os.path.join(path, filename))

    return data_matrix, labels


def load_dating_dataset(filename):
    """ 加载约会数据集 """
    with open(filename, encoding="utf-8-sig") as fr:
        lines = fr.readlines()
    num_lines = len(lines)
    data_matrix = np.zeros((num_lines, 3))  # 三个特征
    labels = []

    for index, line in enumerate(lines):
        line = line.strip().split('\t')
        data_matrix[index, :] = list(map(float, line[0:3]))
        labels.append((line[-1]))
    return data_matrix, labels
