import numpy as np

# 加载 label.npy 文件
labels = np.load('data.npy')

# 打印维度信息
print(labels.shape)


import numpy as np

# 加载label.npy文件
label_data = np.load('label.npy')

# 输出前10个具体值
print("First 10 values in label.npy:", label_data[:10])
