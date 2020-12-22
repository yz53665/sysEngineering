import copy
import random
import matplotlib.pyplot as plt

def plotArraty(array):
    # 显示图像，行代表时间，列代表序列
    camp = plt.get_cmap('Blues')
    plt.imshow(array, camp, interpolation='none')
class AutoCell:
    def __init__(self, ruleNum, arrayLenth, arrayMode=1)
        self.rule = self.initRule(ruleNum)
        self.array = self.initArray(arrayLenth, arrayMode)

    def initRule(self, ruleNum):
        ruleKey = ['111', '110', '101', '100', '011', '010', '001', '000']
        binRule = bin(ruleNum)[2:]
        binRule = str.zfill(binRule, 8)
        rule = {ruleKey[index]:i for index, i in enumerate(binRule)}
        return rule

    def initArray(self, arrayLenth, mode=1):
        '''
        生成初始化序列
        *************paramaters*************
        arrayLenth: 生成的序列长度 
        mode: 序列生成模式  0-完全随机，1-中间一个元素为1，其它为0
        '''
        if mode == 1:
            array = [0 for i in range(arrayLenth)]
            array[int(arrayLenth/2)] = 1
        elif mode == 0:
            array = [random.randint(0,1) for _ in range(arrayLenth)]

        return array

    def run(self, time):
        '''
        根据给定迭代次数运行元胞自动机并显示图像
        *************paramaters*************
        time: 迭代次数
        '''
        iterArray = [self.array]
        for t in range(int(time)):
            x = self.array
            y = copy.copy(self.array)
            for i in range(1, len(x)-1):
                y[i] = rule[str(x[i-1])+str(x[i])+str(x[i+1])]
            iterArray.append(y)
            self.array = y

        plotArraty(iterArray)
        plt.show()

if __name__ == '__main__':
    arrayList = input('请输入初始列表长度:')
    ruleNum = input('请输入规则数:')
    time = input('请输入迭代次数:')
    x = [random.randint(0, 1)]
    rule = {'111': 0, '110': 0, '101': 1, '100': 1, '011': 0, '010': 0,
            '001': 1, '000': 0}
    autoCell(x, rule, time)
