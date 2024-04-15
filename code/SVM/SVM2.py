'''
@File    :     SVM2.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/7 16:59   
@Author        huahai2022
@Desciption
'''
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data=load_iris()
X=data['data']
Y=data['target']
X = X[:, :2]
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


clf=svm.SVC(C=0.8,kernel='rbf',gamma=10,decision_function_shape='ovr')
clf.fit(x_train,y_train)





# 分别打印训练集和测试集的准确率 score(x_train, y_train)表示输出 x_train,y_train在模型上的准确率
def print_accuracy(clf, x_train, y_train, x_test, y_test):
	print('training prediction:%.3f' % (clf.score(x_train, y_train)))
	print('test data prediction:%.3f' % (clf.score(x_test, y_test)))
	# 计算决策函数的值 表示x到各个分割平面的距离
	print('decision_function:\n', clf.decision_function(x_train)[:2])


def draw(clf, x):
	iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
	# 开始画图
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
	# 生成网格采样点
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]

	grid_test = np.stack((x1.flat, x2.flat), axis=1)
	print('grid_test:\n', grid_test[:2])
	# 输出样本到决策面的距离
	z = clf.decision_function(grid_test)
	print('the distance to decision plane:\n', z[:2])
	grid_hat = clf.predict(grid_test)
	# 预测分类值 得到[0, 0, ..., 2, 2]
	print('grid_hat:\n', grid_hat[:2])
	# 使得grid_hat 和 x1 形状一致
	grid_hat = grid_hat.reshape(x1.shape)
	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 能够直观表现出分类边界

	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(Y), edgecolor='k', s=50, cmap=cm_dark)
	plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)
	plt.xlabel(iris_feature[0], fontsize=20)
	plt.ylabel(iris_feature[1], fontsize=20)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.title('Iris data classification via SVM', fontsize=30)
	plt.grid()
	plt.show()


print('-------- eval ----------')
print_accuracy(clf, x_train, y_train, x_test, y_test)
print('-------- show ----------')
draw(clf, X)