'''
@File    :     linear_regression.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/6 21:27   
@Author        huahai2022
@Desciption
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#加载波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"	#波士顿数据集网站
#sep是正则表达式，匹配任意空白字符;前22行是数据集介绍，进行跳过;没有列标题,header=None
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
print(data.shape)
target = raw_df.values[1::2, 2]
print(target.shape)
#将数据集拆分为训练集和测试集
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
model= LinearRegression(fit_intercept=True)
model.fit(train_data, train_target)
y_pred = model.predict(test_data)
#计算均方差损失
print(f"均方差损失(MSE): {mean_squared_error(test_target, y_pred)}")

# 查看回归系数（斜率）
print("回归系数（斜率）:")
print(model.coef_)

# 查看截距
print("截距:")
print(model.intercept_)
