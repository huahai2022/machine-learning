'''
@File    :     SVM3.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/7 17:17   
@Author        huahai2022
@Desciption
'''
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# 创建SVM模型
svm_model = SVC(kernel='rbf')

# 在训练集上拟合SVM模型
svm_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)
