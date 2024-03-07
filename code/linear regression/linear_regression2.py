'''
@File    :     linear_regression2.py    
@Contact :     zhangzhilong2022@gmail.com
@Modify Time   2024/3/6 22:16   
@Author        huahai2022
@Desciption
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"	#波士顿数据集网站
#sep是正则表达式，匹配任意空白字符;前22行是数据集介绍，进行跳过;没有列标题,header=None
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
print(data.shape)
target = raw_df.values[1::2, 2]
print(target.shape)

# 数据标准化
scaler = StandardScaler()
data = scaler.fit_transform(data)
#将数据集拆分为训练集和测试集
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)


# 转换为张量
X_train = torch.Tensor(train_data)
X_label = torch.Tensor(train_target)
y_train = torch.Tensor(test_data)
y_label = torch.Tensor(test_target)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(13, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    print(outputs.shape)
    loss = criterion(outputs, X_label)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 打印每个epoch的损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
with torch.no_grad():
    predicted = model(y_train)
    print('Predicted values:', predicted)
    #均方差损失
    mse = criterion(predicted, y_label)
    print('Mean Squared Error:', mse.item())