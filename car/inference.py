'''
Extended by Peggy Wang @PeggyYuchunWang
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, math, random, collections


# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):

    NUM_PARTICLES = 500

    # Function: Init
    # --------------
    # Constructor that initializes an ParticleFilter object which has
    # (numRows x numCols) number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in an integer-valued defaultdict.
        # Use self.transProbDict[oldTile][newTile] to get the probability of transitioning from oldTile to newTile.
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly.
        self.particles = collections.defaultdict(int)
        potentialParticles = self.transProbDict.keys()
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a defaultdict from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    ##################################################################################
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # This algorithm takes two steps:
    # 1. Re-weight the particles based on the observation.
    # 2. Re-sample the particles.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SENSOR_STD
    #
    ##################################################################################
    def observe(self, agentX, agentY, observedDist):
        distDict = {}
        for (r, c) in self.particles:
            numPar = self.particles[(r, c)]
            x = util.colToX(c)
            y = util.rowToY(r)
            d = math.sqrt((x - agentX)**2 + (y - agentY)**2)
            emissionProb = util.pdf(d, Const.SENSOR_STD, observedDist)
            distDict[(r, c)] = emissionProb * float(numPar)/float(self.NUM_PARTICLES)
        newParticles = collections.defaultdict(int)
        for i in range(self.NUM_PARTICLES):
            newParticles[util.weightedRandomChoice(distDict)] += 1
        self.particles = newParticles

        self.updateBelief()

    ##################################################################################
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Reads |self.particles|, representing particle locations at time $t$, and
    # writes an updated |self.particles| with particle locations at time $t+1$.
    #
    # This algorithm takes one step:
    # 1. Proposal based on the particle distribution at current time $t$.
    ##################################################################################
    def elapseTime(self):
        newParticles = collections.defaultdict(int)
        for oldTile in self.particles:
            weightDict = collections.defaultdict(int)
            numPar = self.particles[oldTile]
            if oldTile not in self.transProbDict:
                continue
            for newTile in self.transProbDict[oldTile]:
                weightDict[newTile] = self.transProbDict[oldTile][newTile] * float(numPar)/float(self.NUM_PARTICLES)
            for i in range(numPar):
                newParticles[util.weightedRandomChoice(weightDict)] += 1
        self.particles = newParticles

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile.
    # Belief probabilities should sum to 1.
    def getBelief(self):
        return self.belief
