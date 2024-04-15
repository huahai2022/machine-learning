'''
@File    :     logistic regression.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/8 17:47   
@Author        huahai2022
@Desciption
'''
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
# 加载sklearn内置的iris鸢尾花数据集
iris = datasets.load_iris()

X = iris.data
y = iris.target

print('Sample num: ', len(y))#150

# 将原始数据集随机切分成两部分，作为训练集和测试集，其中测试集占总样本30%。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置模型参数并使用训练集训练模型。
clf = LogisticRegression(C=1.0, penalty='l2',tol=1e-6)
print(clf)
# 训练模型
clf.fit(X_train, y_train)

# 使用测试集预测结果
ans = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test,ans))
# 100%