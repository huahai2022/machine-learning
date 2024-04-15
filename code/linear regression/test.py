import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# 生成 x 值
x = np.linspace(-10, 10, 100)

# 计算 relu 函数值
y = relu(x)

# 绘制图像
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Function')
plt.grid(True)
plt.show()