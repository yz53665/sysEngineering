import itertools
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def ReadData(path):
    return pd.read_csv(path, header=None)

def ManageIrisData(src, trainSize=0.7):
    xPrime = src[list(range(4))]
    y = pd.Categorical(src[4]).codes
    xPrimeTrain, xPrimeTest, yTrain, tTest = train_test_split(xPrime, y, train_size=0.7, random_state=0)
    return xPrimeTrain, xPrimeTest, yTrain, tTest

def ChooseClassifyFeatures(ftrs, train, test):
    train = train[ftrs]
    test = test
    return train, test

'''
准备iris数据
param:
    path:数据地址
    ftrs:选择参加训练的特征，list
return:
    xTrain:用于训练的数据
    xTest:用于测试的数据
    yTrain:训练数据的标签
    yTest:测试数据的标签
'''
def PrepareData(path, ftrs):
    src = ReadData(path)
    xPrimeTrain, xPrimeTest, yTrain, yTest = ManageIrisData(src)
    xTrain, xTest = ChooseClassifyFeatures(ftrs, xPrimeTrain, xPrimeTest)
    return xTrain, xTest, yTrain, yTest

'''
准备决策树模型
param:
    trainSet:训练数据
    testSet:测试数据
return:
    model:实例化的模型
'''
def PrepareDcTreeModule(trainSet, testSet):
    model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    model.fit(trainSet, testSet)
    return model

'''
运行模型
param:
    model:待运行的模型
    x:待预测数据
    y:预测数据对应的实际标签
return:
    accuracy:预测的准确率
'''
def RunModel(model, x, y):
    yPred = model.predict(x)
    accuracy = accuracy_score(yPred, y)
    return accuracy

'''
打印预测结果
param:
    acc:要打印的准确率
    title:准确率所属类（测试集、训练集）
'''
def PrintResult(acc, title):
    print('\t {} accuracy: {:.4f} %'.format(title, 100 * acc))


'''
用于根据给定的model绘制决策平面
'''
class DcisionSurface:
    def __init__(self, model, labels):
        self.model = model 
        self.labels = labels
        self.xRanges = []
        self.yRanges = []
        self.xxs = []
        self.yys = []
        self.zs = []
        self.numLabels = len(labels)
        self.ftrsName= []
        self.ftrs = []

    '''
    根据输入的列表产生两两配对的组合数
    param:
        numLabels:分类数量
        ftrs:想要绘制的特征标号，list
        ftrsName:特征名称
    '''
    def AddFeaturesGroup(self, ftrs, ftrsName):
         self.ftrs = list(itertools.combinations(ftrs, 2))
         self.ftrsName = ftrsName

    '''
    添加需要绘制的特征向量的最大最小范围
    param:
        data:模型的输入输入数据
    '''
    def AddRange(self, data):
        for i in self.ftrs:
            X = data[:, i]
            xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
            yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
            self.xRanges.append((xMin, xMax))
            self.yRanges.append((yMin, yMax))

    '''
    根据给定的x，y坐标范围计算范围内model的取值平面
    param:
        model:想要求取值平面的模型
        xRange:取值平面的x坐标范围
        yRange:取值平面的y坐标范围
        step:步长
    return:
        Z:坐标上每点取值
    '''
    def GetPointsValue(self, data, dataLabels, step):
        self.AddRange(data)
        for xRange, yRange, i in zip(self.xRanges, self.yRanges, self.ftrs):
            xx, yy = np.meshgrid(np.arange(xRange[0], xRange[1], step),
                                 np.arange(yRange[0], yRange[1], step))
            self.model.fit(data[:, i], dataLabels)
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            self.xxs.append(xx)
            self.yys.append(yy)
            self.zs.append(Z)

    '''
    绘制决策平面
    '''
    def DrawSurface(self):
        plt.figure()
        for index, i in enumerate(self.ftrs):
            xid, yid = i
            plt.subplot((len(self.ftrs)) / 3 + 1, 3, index + 1)
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            cs = plt.contourf(self.xxs[index], self.yys[index], self.zs[index], cmap=plt.cm.RdYlBu)
            plt.xlabel(self.ftrsName[xid])
            plt.ylabel(self.ftrsName[yid])

    '''
    在决策平面上绘制数据的散点图
    param:
        data:输入模型的数据集
        dataLabels:对应数据的标签
        plotColors:散点的颜色，string，要与分类数量一致，如'bgr'
    '''
    def DrawScatter(self, data, dataLabels,  plotColors):
        for index, i in enumerate(self.ftrs):
            xid, yid = i
            X = data[:, i]
            plt.subplot((len(self.ftrs)) / 3 + 1, 3, index + 1)
            for index, color in zip(range(self.numLabels), plotColors):
                idx = np.where(dataLabels == index)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=self.labels[index], cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    '''
    绘制决策树
    '''
    def DrawDCTree(self):
        plt.figure(figsize=(6, 8), dpi=100)
        plot_tree(self.model, filled=True)

'''
绘制给定data中每一列的取值范围并以列表的形式返回
'''
def GetRange(data):
    ranges = []
    for i in range(data.shape[1]):
        minumum, maximum = data[:, i].min(), data[:, i].max()
        ranges.append((minumum, maximum))
    return ranges 

'''
根据GetRange提供的数据范围生成categories * dataNum个随机数字
param:
    ranges:
    categories:
    dataNum:
return:
    np.array:前categories列为数据，最后一列为labels
'''
def CreateRandomData(ranges, categories, dataNum):
    datas = []
    for minumum, maximum in ranges:
        data = np.random.uniform(minumum, maximum, dataNum)
        datas.append(data)
    datas = np.asarray(datas)
    datas = datas.transpose()
    return datas

if __name__ == '__main__':
    path = 'iris.data'
    randpath = 'randData.csv'
    ftrsName = ['sepal length', 'sepal width', 'petal length', 'petal width']
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    # iris数据集
    xTrain, xTest, yTrain, yTest = PrepareData(path, [0, 1, 2, 3])
    model = PrepareDcTreeModule(xTrain, yTrain)
    accTrain = RunModel(model, xTrain, yTrain)
    accTest = RunModel(model, xTest, yTest)
    PrintResult(accTrain, 'Train')
    PrintResult(accTest, 'Test')

    x = np.vstack((xTrain, xTest))
    y = np.hstack((yTrain, yTest))

    # 随机生成100个数据
    length = 100
    ranges = GetRange(x)
    randData= CreateRandomData(ranges, 4, length)

    yPreRandInt = model.predict(randData)
    yPreRandStrs = []
    for index in range(length):
        yPreRandStr = labels[yPreRandInt[index]]
        yPreRandStrs.append(yPreRandStr)
    yPreRandStrs = np.asarray(yPreRandStrs)
    yPreRandStrs = yPreRandStrs.reshape((-1,1))
    output = np.hstack((randData, yPreRandStrs))
    # 预测结果存入randPath中
    df = pd.DataFrame(output, columns=ftrsName+['predict'])
    df.to_csv(randpath, header=0)

    # 下面开始绘制决策平面
    # 初始化类
    painter = DcisionSurface(model, labels)
    painter.AddFeaturesGroup(list(range(4)), ftrsName)
    painter.GetPointsValue(x, y, 0.02)
    # 绘制决策平面
    painter.DrawSurface()
    # 绘制散点图
    painter.DrawScatter(x, y, 'ryb')
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    # 绘制决策树
    painter.DrawDCTree()
    plt.show()

    # 绘制随机生成的100个数据在决策平面上的散点图
    plt.figure()
    painter.DrawSurface()
    painter.DrawScatter(randData, yPreRandInt, 'ryb')
    plt.show()
