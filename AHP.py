import numpy as np 

Z = np.array([
        [1,2,7,5,5],
        [1/2,1,4,3,3],
        [1/7,1/4,1,1/2,1/3],
        [1/5,1/3,2,1,1],
        [1/5,1/3,3,1,1]
        ])

A1 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

A2 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

A3 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

A4 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

A5 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

# 提取特征值和特征向量，复数形式
eigenValues, eigenVectors = np.linalg.eig(matrix)
print(eigenValues)
print()
print(eigenVectors)

# 获取最大特征值id
max_idx = np.argmax(eigenValues)
print(max_idx)

# 获取最大特征值
max_eigen = eigenValues[max_idx].real
print(max_eigen)

# 获取最大特征值对应的特征向量
eigen = eigenVectors[:, max_idx].real
print(eigen)

# 对特征向量进行归一化处理
eigen = eigen / eigen.sum()
print(eigen)
