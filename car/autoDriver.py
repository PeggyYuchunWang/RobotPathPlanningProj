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
# A modified autonomous driver that drives around a map, avoiding cars
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
        self.pq = []

    def getTerminalState(self, agentGraph):
        for n in agentGraph.nodeMap:
            if agentGraph.isTerminal(n):
                self.terminalState = n
                print self.terminalState
                break

    def distanceHeuristic(self, agentGraph, nodeId):
        if self.terminalState == None:
            return -1
        terminalX = agentGraph.getNodeX(self.terminalState)
        terminalY = agentGraph.getNodeY(self.terminalState)
        nodeX = agentGraph.getNodeX(nodeId)
        nodeY = agentGraph.getNodeY(nodeId)
        dist = (terminalX - nodeX) **2 + (terminalY - nodeY) ** 2
        return dist

    def combinedHeuristic(self, beliefOfOtherCars, agentGraph, nodeId):
        distance = self.distanceHeuristic(agentGraph, nodeId)
        print(distance)
        probability = self.nodeCarProb(beliefOfOtherCars, agentGraph, nodeId)
        return distance + 1000000 * probability

    # Function: Get Autonomous Actions
    # ---------------------
    # Given the current belief about where other cars are and a graph of how
    # one can drive around the world, chose a next action.
    def getAutonomousActions(self, beliefOfOtherCars, agentGraph):
        if self.terminalState == None:
            self.getTerminalState(agentGraph)
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
        driveForward = not self.isCloseToOtherCar(beliefOfOtherCars)
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        # if driveForward:
        actions[Car.DRIVE_FORWARD] = 1.0
        return actions

    # Funciton: Is Close to Other Car
    # ---------------------
    # Given the current belief about where other cars are decides if
    # there is a car in the spot where we are about to drive.
    def isCloseToOtherCar(self, beliefOfOtherCars):
        offset = self.dir.normalized() * 1.5 * Car.LENGTH
        newPos = self.pos + offset
        row = util.yToRow(newPos.y)
        if row >= beliefOfOtherCars.getNumRows():
            row = beliefOfOtherCars.getNumRows() - 1
        col = util.xToCol(newPos.x)
        if col >= beliefOfOtherCars.getNumCols():
            col = beliefOfOtherCars.getNumCols() - 1
        p = beliefOfOtherCars.getProb(row, col)
        return p > AutoDriver.THRESHOLD_PROB

    # Funciton: Is Node Close to Other Car
    # ---------------------
    # Given the current belief about where other cars are decides if
    # there is a car in the spot where we are about to drive.
    def isNodeCloseToOtherCar(self, beliefOfOtherCars, agentGraph, nodeId):
        newPos = agentGraph.getNodePos(nodeId)
        row = util.yToRow(newPos.y)
        if row >= beliefOfOtherCars.getNumRows():
            row = beliefOfOtherCars.getNumRows() - 1
        col = util.xToCol(newPos.x)
        if col >= beliefOfOtherCars.getNumCols():
            col = beliefOfOtherCars.getNumCols() - 1
        p = beliefOfOtherCars.getProb(row, col)
        return p > AutoDriver.MIN_PROB

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
            self.pq = []
            for n in nextIds:
                heuristicVal = self.combinedHeuristic(beliefOfOtherCars, agentGraph, n)
                # print(heuristicVal)
                heapq.heappush(self.pq, (heuristicVal, n))
            self.nextId = heapq.heappop(self.pq)[1]
        # if nextIds == []:
        #     self.nextId = None
        # else:
        #     minDist = float("inf")
        #     self.nextId = None
        #     for n in nextIds:
        #         tempDist = self.distanceHeuristic(agentGraph, n)
        #         if tempDist < minDist and not self.isNodeCloseToOtherCar(beliefOfOtherCars, agentGraph, n):
        #             minDist = tempDist
        #             self.nextId = n
        #     if self.nextId == None:
        #         if self.isCloseToOtherCar(beliefOfOtherCars):
        #             self.nextId = self.nodeId
        #         else:
        #             minProb = float("inf")
        #             for n in nextIds:
        #                 tempProb = self.nodeCarProb(beliefOfOtherCars, agentGraph, n)
        #                 if tempProb < minProb:
        #                     minProb = tempProb
        #                     self.nextId = n
