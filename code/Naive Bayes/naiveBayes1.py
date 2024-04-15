'''
@File    :     naiveBayes1.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/8 10:48   
@Author        huahai2022
@Desciption
'''
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
# 加载MNIST数据集
digits = datasets.load_digits()
X = digits.data
print(X.shape)
y = digits.target
print(y.shape)

plt.imshow(X[0].reshape(8, 8),cmap='gray')
plt.show()
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#创建高斯贝叶斯分类器
model = BernoulliNB()
model.fit(X_train, y_train)
#预测分类结果
prediction=model.predict(X_test)
print(f"分类准确率为{accuracy_score(y_test,prediction)}")
