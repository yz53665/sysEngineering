class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        if self.isEmpty():
            pritn('stack is already empty!')
            return
       x = self.items[-1]
       del self.items[-1]
       return x
    
    def empty(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0
