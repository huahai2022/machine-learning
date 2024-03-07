import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据集
def generate_dataset():
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + np.random.randn(100, 1)  # y = 2X + 噪音
    return X, y

# 训练线性回归模型
def train_linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置项
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # 最优参数的闭式解
    return theta_best

# 预测
def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置项
    y_pred = X_b.dot(theta)
    return y_pred

# 生成数据集
X, y = generate_dataset()

# 训练线性回归模型
theta_best = train_linear_regression(X, y)

# 预测
X_new = np.array([[0], [10]])  # 预测新数据点
y_pred = predict(X_new, theta_best)

# 绘制数据集和线性回归模型的拟合线
plt.scatter(X, y)
plt.plot(X_new, y_pred, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.grid(True)
plt.show()