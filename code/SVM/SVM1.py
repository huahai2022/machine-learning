from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data=load_iris()
X=data['data']
Y=data['target']
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

clf=svm.SVC(C=0.8,kernel='rbf',decision_function_shape='ovr')
clf.fit(x_train,y_train)

#计算测试集的准确率
print(clf.score(x_test,y_test))