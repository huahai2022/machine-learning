'''
@File    :     knnClassifier.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/2/29 18:02   
@Author        huahai2022
@Desciption	   KNN算法实现正态分布数据的分类
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as  KNN
# 创建数据
def createDatabase():
    # 第一组数据，loc表示横坐标均值为5，纵坐标均值为2，scale表示标准差为0.5，szie表示生成100个样本，每个样本里面有两个维度，分别是横坐标和纵坐标。
    data1 = np.random.normal(loc=[5, 2], scale=1, size=(100, 2))
    data2 = np.random.normal(loc=[5, -2], scale=1, size=(100, 2))
    data3 = np.random.normal(loc=[2, 0], scale=0.8, size=(100, 2))
    data=np.concatenate((data1,data2,data3),axis=0)
    labels = [1] * 100 + [2] * 100 + [3] * 100
    return data,labels
def showDatabase(data1,data2,data3):
    # 可视化数据
    plt.scatter(data1[:, 0], data1[:, 1], color='red', label='Group 1')
    plt.scatter(data2[:, 0], data2[:, 1], color='blue', label='Group 2')
    plt.scatter(data3[:, 0], data3[:, 1], color='green', label='Group 3')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Visualization')
    # 显示数据的标签
    plt.legend()
    plt.show()


def normalDatabase(data):
    # 计算数据的最小值和最大值
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    print(data_max)
    print(data_min)
    # 对 x 维度进行归一化
    xNormalized = (data[:, 0] - data_min[0]) / (data_max[0] - data_min[0])
    # 对 y 维度进行归一化
    yNormalized = (data[:, 1] - data_min[1]) / (data_max[1] - data_min[1])
    # 将 x 和 y 归一化后的数据合并为一个数组
    xyNormalized = np.array(list(zip(xNormalized, yNormalized)))
    return xyNormalized,data_min,data_max
def knnClassify(data,labels):
    knnModel=KNN(n_neighbors=10)
    knnModel.fit(data,labels)
    return knnModel
if __name__=='__main__':
    #创建训练集
    data,labels=createDatabase()
    #展示训练集
    showDatabase(data[:100,:],data[100:200,:],data[200:300,:])
    #归一化训练集
    noramlData,minData,maxData=normalDatabase(data)
    #展示归一化后的训练集
    showDatabase(noramlData[:100,:],noramlData[100:200,:],noramlData[200:300,:])
    #创建分类器
    knnModel=knnClassify(data=noramlData,labels=labels)
    #创建测试数据(10,0),(2,2)并进行归一化
    test=[(10,0),(2,2)]
    normalTest=[((10-minData[0])/(maxData[0]-minData[0]),(0-minData[1])/(maxData[1]-minData[1])),((1-minData[0])/(maxData[0]-minData[0]),(1-minData[1])/(maxData[1]-minData[1]))]
    #对测试数据进行预测
    prediction=knnModel.predict(test)
    print(f"{test[0]}属于类别{prediction[0]};{test[1]}属于类别{prediction[1]}")
    #展示预测结果所属类别
    preProba=knnModel.predict_proba(test)
    print(f"{test[0]}属于类别1的概率为{[preProba[0][0]]},属于类别2的概率为{[preProba[0][1]]},属于类别3的概率为{[preProba[0][2]]}")
    print(f"{test[1]}属于类别1的概率为{[preProba[1][0]]},属于类别2的概率为{[preProba[1][1]]},属于类别3的概率为{[preProba[1][2]]}")
    #创建测试集，为训练集数据量的一半
    testData,testLabels=createDatabase()
    testCutData = np.concatenate((testData[:50, :], testData[100:150, :], testData[200:250, :]), axis=0)
    testCutLabels=np.concatenate((testLabels[:50],testLabels[100:150],testLabels[200:250]),axis=0)
    #对测试集数据进行归一化处理
    xNormal=(testCutData[:,0]-minData[0])/(maxData[0]-minData[0])
    yNormal=(testCutData[:,1]-minData[1])/(maxData[1]-minData[1])
    xyNormal=list(zip(xNormal,yNormal))
    #对测试集数据进行预测
    scores=knnModel.score(xyNormal,testCutLabels)
    print(f"测试集的准确率为{scores}")