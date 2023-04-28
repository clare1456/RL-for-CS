'''
Author: 71
LastEditTime: 2023-04-15 12:15:46
version: 
Description: solution class defination
'''


class VRPSolution():
    def __init__(self) -> None:
        self.objVal = 0 # solution obj
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
        self.objVal = objVal # solution obj
        self.path = path
        self.travelTime = travelTime
        self.travelTimeList = travelTimeList
        self.distance = distance
        self.load = load
        self.loadList = loadList

