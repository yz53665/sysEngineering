import copy
time = input()
x = [0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1]
rule = {'111':0, '110':0, '101':1, '100':1, '011':0, '010':0, '001':1, '000':0}
for t in range(int(time)):
    y = copy.copy(x)
    for i in range(1,len(x)-1):
        y[i] = rule[str(x[i-1])+str(x[i])+str(x[i+1])]
    x = y
print(x)
