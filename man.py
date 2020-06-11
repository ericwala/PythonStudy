class Man:
    def __init__(self,name):
        self.name = name
        print("Initilized!")
    def hello(self):
        print("hello",self.name,"!")
    def goodbye(self):
        print("goodbye",self.name,"!")
        
m = Man("David")
m.hello()
m.goodbye()
