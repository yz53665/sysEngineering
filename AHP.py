import numpy as np 

def AHP(matrixs):
    normWeights = []
    CRs = []
    CIs = []
    RIs = []
    for index, i in enumerate(matrixs):
        print('第' + str(index) + '个')
        # 提取特征值和特征向量，复数形式
        eigenValues, eigenVectors = np.linalg.eig(i)

        # 获取最大特征值id
        max_idx = np.argmax(eigenValues)

        # 获取最大特征值
        max_eigen = eigenValues[max_idx].real
        print('最大特征值')
        print(max_eigen)

        # 获取最大特征值对应的特征向量
        eigen = eigenVectors[:, max_idx].real
        print('最大特征值对应特征向量')
        print(eigen)

        # 对特征向量进行归一化处理
        eigen = eigen / eigen.sum()
        normWeights.append(eigen)

        # 判断一致性
        CI = (max_eigen - i.shape[0]) / (i.shape[0] - 1)
        CIs.append(CI)

        RI = 1
        if i.shape[0] == 4:
            RI = 0.9
        elif i.shape[0] == 5:
            RI = 1.12
        RIs.append(RI)

        CR = CI / RI
        print('随机一致性比例：')
        print(CR)

    A = normWeights[0]
    B = np.asarray([normWeights[i+1] for i in range(len(matrixs) - 1)])
    result = np.dot(A, B)
    
    # 层次总排序一致性检验
    CRAll = np.dot(A, np.asarray(CIs[1::])) / np.dot(A, np.asarray(RIs[1::]))
    print('层次总排序一致性检验：')
    print(CRAll)

    return CRs, normWeights, result

Z = np.array([
        [1,2,7,5,5],
        [1/2,1,4,3,3],
        [1/7,1/4,1,1/2,1/3],
        [1/5,1/3,2,1,1],
        [1/5,1/3,3,1,1]
        ])

A1 = np.array([
    [1,1/3,1/5,1/7],
    [3,1,1/2,1/4],
    [5,2,1,1/2],
    [7,4,2,1]
    ])

A2 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

A3 = np.array([
    [1,6,5,8],
    [1/6,1,1,2],
    [1/5,1,1,7],
    [1/8,1/2,1/7,1]
    ])

A4 = np.array([
    [1,3,5,7],
    [1/3,1,3,5],
    [1/5,1/3,1,3],
    [1/7,1/5,1/3,1]
    ])

A5 = np.array([
    [1,1,1/3,1/3],
    [1,1,1/2,1/5],
    [3,2,1,1],
    [3,5,1,1]
    ])

matrix = [Z, A1, A2, A3, A4, A5]
result = AHP(matrix)

print('总目标权值：')
print(result[2])
print('目标权值之和：')
print(np.sum(result[2]))
