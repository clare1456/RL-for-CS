
class VRPSolution():
    def __init__(self) -> None:
        self.objVal = 0 
        self.vehicleNum = 0
        self.pathSet = {}
        self.solSet = {}
        self.distance = {}
        self.travelTime = {}
        self.totalLoad = {}
        self.pathNum = []
        
          
class SPPSolution():
    def __init__(self,objVal=0,path=[],travelTime=0,travelTimeList=[],distance=0,load=0,loadList=[]) -> None:
        self.status = False
        self.objVal = objVal 
        self.path = path
        self.travelTime = travelTime
        self.travelTimeList = travelTimeList
        self.distance = distance
        self.load = load
        self.loadList = loadList

