'''
@File    :     decisionTree.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/4 18:16   
@Author        huahai2022
@Desciption	   决策树算法实现
'''
import math
import operator


def calculate_entropy(dataset):
    # 使用class_counts记录各个标签样本的数量
    class_counts = {}
    for data in dataset:
        label = data[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    entropy = 0
    total_samples = len(dataset)
    # 计算数据集的熵
    for count in class_counts.values():
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
    return entropy


def createDataset():
    myDataset = [[1, 2, 2, 0],
                 [2, 1, 1, 0],
                 [2, 1, 3, 1],
                 [3, 1, 3, 1],
                 [3, 3, 2, 0],
                 [2, 2, 3, 1],
                 [1, 2, 2, 0],
                 [1, 1, 1, 0],
                 [3, 2, 3, 1],
                 [3, 3, 2, 1]]
    # 最后一列表示标签，0表示未购买产品，1表示购买产品
    labels=["年龄", "收入", "学历", "是否购买产品"]
    return myDataset,labels


def calculate_condition_entropy(dataset, feature_index):
    feature_counts = {}  # 用于统计各个特征以及对应标签的出现次数
    for row in dataset:
        feature = row[feature_index]
        label = row[-1]
        if feature not in feature_counts:  # 初始化为字典的嵌套，0表示未购买产品，1表示购买产品
            feature_counts[feature] = {0: 0, 1: 0}
        feature_counts[feature][label] += 1
    # 得到对应特征的统计信息，比如第二列的统计信息为{2: {0: 2, 1: 2}, 1: {0: 2, 1: 2}, 3: {0: 1, 1: 1}}

    entropy = 0
    dataset_len = len(dataset)  # 数据集的总长度
    for feature in feature_counts:
        counts = sum(feature_counts[feature].values())  # 统计特征出现的次数
        conditional_entropy = 0
        probability = counts / dataset_len  # 计算特征出现的概率
        for label in feature_counts[feature]:
            conditional_probability = feature_counts[feature][label] / counts  # 在特征固定时，用来计算标签的概率
            conditional_entropy -= conditional_probability * math.log2(
                conditional_probability) if conditional_probability != 0 else 0  # 计算条件熵
        entropy += probability * conditional_entropy  # 条件熵的加权平均
    return entropy  # 返回总熵值


def split_dataset(dataset, feature_index, value):
    """
    :param dataset: 数据集
    :param feature_index: 选择的特征
    :param value: 特征的label
    :return: 切分后的数据集
    """
    retDataset = []
    for row in dataset:
        if row[feature_index] == value:  # 选择特征，如果特征是某个值的话，就把该特征以及该特征后面的特征组成新的子集
            subDataset = row[:feature_index]
            subDataset.extend(row[feature_index + 1:])
            retDataset.append(subDataset)
    return retDataset


def choose_best_feature_to_split(dataset):
    """
    :param dataset: 数据集
    :return: 最佳特征的索引
    """
    feature_dim = len(dataset[0]) - 1  # 数据集中的特征维度
    entroy = calculate_entropy(dataset)  # 计算数据集的熵
    best_feature_gain = {}  # 使用字典存储特征的增益
    for i in range(feature_dim):  # 计算每个特征的信息增益
        gain = calculate_entropy(dataset) - calculate_condition_entropy(dataset, i)
        best_feature_gain[i] = gain
    max_value = max(best_feature_gain.values())  # 返回最大的信息增益的维度
    max_keys = [key for key, value in best_feature_gain.items() if value == max_value]
    return max_keys[0]


def majority_cnt(class_list):
    # 返回类别中出现次数最多的类别
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_decision_tree(dataset,labels,feature_labels):
    """
   :param dataset: 数据集
    :return: 决策树
    """
    class_list = [row[-1] for row in dataset]  # 获取数据集的类别
    if class_list.count(class_list[0]) == len(class_list):  # 如果类别完全相同则停止划分
        return class_list[0]
    if len(dataset[0]) == 1:  # 如果所有特征都遍历完了，则停止划分
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    labels.remove(best_feature_label)	#去除已经使用过的标签
    feature_labels.append(best_feature_label)
    my_tree={best_feature_label:{}}
    feature_values=set([row[best_feature] for row in dataset])
    for value in feature_values:
        sub_labels=labels[:]
        sub_dataset=split_dataset(dataset,best_feature,value)
        my_tree[best_feature_label][value]=create_decision_tree(sub_dataset,sub_labels,feature_labels)	#递归构建子树
    return my_tree
def predict(decision_tree, sample):
    if isinstance(decision_tree, dict):
        for feature in decision_tree:
            value = sample.get(feature)
            if value in decision_tree[feature]:
                subtree = decision_tree[feature][value]
                return predict(subtree, sample)
        return None
    else:
        return decision_tree
if __name__ == '__main__':
    myDataset,labels = createDataset()
    decision_tree=create_decision_tree(myDataset,labels,[])
    print(decision_tree)
    test_dataset =[{'学历': 2, '年龄': 3, '收入': 3},{'学历': 3, '年龄': 1, '收入': 3}]
    print(f"{test_dataset[0]}的决策结果为{predict(decision_tree, test_dataset[0])}")
    print(f"{test_dataset[1]}的决策结果为{predict(decision_tree, test_dataset[1])}")
