'''
Extended by Peggy Wang @PeggyYuchunWang
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.model.car.junior import Junior
from engine.model.car.car import Car

import util
import random
import heapq

# Class: AutoDriver
# ---------------------
# An initially naive autonomous driver that drives around a map, avoiding cars
# based on beliefs. Feel free to extend this class. It is *not* required!
class AutoDriver(Junior):

    MIN_PROB = 0.02
    THRESHOLD_PROB = 0.05

    # Funciton: Init
    # ---------------------
    # Create an autonomous driver. Give it a number of heartBeats to wait before
    # it starts to drive
    def __init__(self):
        self.nodeId = None
        self.nextId = None
        self.nextNode = None
        self.burnInIterations = 30
        self.terminalState = None
        self.startState = None
        self.pq = []

    def getTerminalState(self, agentGraph):
        for n in agentGraph.nodeMap:
            if agentGraph.isTerminal(n):
                self.terminalState = n
                print self.terminalState
                break

    def goalHeuristic(self, agentGraph, nodeId):
        if self.terminalState == None:
            return -1
        terminalX = agentGraph.getNodeX(self.terminalState)
        terminalY = agentGraph.getNodeY(self.terminalState)
        nodeX = agentGraph.getNodeX(nodeId)
        nodeY = agentGraph.getNodeY(nodeId)
        dist = (terminalX - nodeX)**2 + (terminalY - nodeY)** 2
        return dist

    def startCost(self, agentGraph, nodeId):
        if self.startState == None:
            return -1
        terminalX = agentGraph.getNodeX(self.startState)
        terminalY = agentGraph.getNodeY(self.startState)
        nodeX = agentGraph.getNodeX(nodeId)
        nodeY = agentGraph.getNodeY(nodeId)
        dist = (terminalX - nodeX)**2 + (terminalY - nodeY)** 2
        return dist

    # Function: Get Autonomous Actions
    # ---------------------
    # Given the current belief about where other cars are and a graph of how
    # one can drive around the world, chose a next action.
    # use A* to calculate optimal path
    def getAutonomousActions(self, beliefOfOtherCars, agentGraph):
        if self.terminalState == None:
            self.getTerminalState(agentGraph)
        if self.startState == None:
            self.startState = self.nodeId
        # Chose a next node to drive towards. Note that you can ask
        # a if its a terminal using node.isTerminal()
        if self.nodeId == None:
            self.nodeId = agentGraph.getNearestNode(self.pos)
        if self.nextId == None:
            self.choseNextId(agentGraph, beliefOfOtherCars)
        if agentGraph.atNode(self.nextId, self.pos):
            self.nodeId = self.nextId
            self.choseNextId(agentGraph, beliefOfOtherCars)

        # given a next node, drive towards that node. Stop if you
        # are too close to another car
        goalPos = agentGraph.getNode(self.nextId).getPos()
        vectorToGoal = goalPos - self.pos
        wheelAngle = -vectorToGoal.get_angle_between(self.dir)
        # driveForward = not self.isCloseToOtherCar(beliefOfOtherCars)
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        actions[Car.DRIVE_FORWARD] = 1.0
        return actions

    def nodeCarProb(self, beliefOfOtherCars, agentGraph, nodeId):
        newPos = agentGraph.getNodePos(nodeId)
        row = util.yToRow(newPos.y)
        if row >= beliefOfOtherCars.getNumRows():
            row = beliefOfOtherCars.getNumRows() - 1
        col = util.xToCol(newPos.x)
        if col >= beliefOfOtherCars.getNumCols():
            col = beliefOfOtherCars.getNumCols() - 1
        p = beliefOfOtherCars.getProb(row, col)
        return p

    # Function: Chose Next Id
    # ---------------------
    # You have arrived at self.nodeId. Chose a next node to drive
    # towards.
    def choseNextId(self, agentGraph, beliefOfOtherCars):
        nextIds = agentGraph.getNextNodeIds(self.nodeId)
        if nextIds == []:
            self.nextId = None
        else:
            self.nextId = random.choice(nextIds)
        path = self.aStar(agentGraph, beliefOfOtherCars)
        if len(path) > 1:
            self.nextId = path[1]
        else:
            self.nextId = self.nodeId

    def aStar(self, agentGraph, beliefOfOtherCars):
        current = self.nodeId
        goal = self.terminalState
        allNodes = agentGraph.getAllNodes()
        newPath = [current]
        pq = []
        heapq.heappush(pq, (self.goalHeuristic(agentGraph, current), newPath))
        seen = set()
        while len(pq) != 0:
            currentPath = heapq.heappop(pq)[1]
            currentState = currentPath[-1]
            if currentState == goal:
                return currentPath
            if currentState in seen:
                continue
            seen.add(currentState)
            for nextState in allNodes:
                if self.nodeCarProb(beliefOfOtherCars, agentGraph, nextState) < self.MIN_PROB:
                    path = currentPath.append(nextState)
                    cost = self.startCost(agentGraph, currentState) + self.goalHeuristic(agentGraph, currentState)
                    heapq.heappush(pq, (self.startCost(agentGraph, currentState) + cost, newPath))
