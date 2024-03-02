'''
@File    :     KNN.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/2/29 12:29   
@Author        huahai2022
@Desciption	   KNN算法原理，简单示例
'''
import math
import numpy as np

# 创建训练数据集
def createDatabse():
	xTrain = [[5.1, 3.5],
			  [4.9, 3.0],
			  [6.7, 3.1],
			  [6.0, 3.0],
			  [5.5, 2.8]]
	yTrain = ['A', 'A', 'B', 'B', 'B']
	return xTrain, yTrain


# 计算点之间的欧几里得距离
def enclideanDistance(x1, x2):
	distance = 0
	for i in range(len(x1)):
		distance += pow((x1[i] - x2[i]), 2)
	return math.sqrt(distance)


# knn算法
def knn(xTrain, yTrain, x_test, k):
	# 计算距离，二维数组，一维度表示训练集下标，二维度表示距离
	distances = []
	for i in range(len(xTrain)):
		dist = enclideanDistance(xTrain[i], x_test)
		distances.append((i, dist))
	# 根据距离排序
	distances.sort(key=lambda x: x[1])
	# 得到前k个邻居，一维度表示训练集的下标，二维度表示距离
	neighbors = distances[:k]
	# 统计邻居所属类别
	counts = {}
	for neighbor in neighbors:
		label = yTrain[neighbor[0]]
		# dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值
		counts[label] = counts.get(label, 0) + 1
	# 对counts进行降序排列
	sortedCount = sorted(counts.items(), key=lambda x: x[1], reverse=True)
	return sortedCount[0][0]


if __name__ == '__main__':
	xTrain, yTrain = createDatabse()
	xTest = [5.7, 3.2]
	yTest=knn(xTrain=xTrain, yTrain=yTrain, x_test=xTest, k=3)
	print(yTest)
