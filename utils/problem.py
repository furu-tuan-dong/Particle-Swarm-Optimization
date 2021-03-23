import numpy as np
import math
import time
import copy
from tqdm.std import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter 

plt.ion()
EXTREME = 10e3

## **Functions**

### F1: Rastrigin

class Rastrigin():
  def __init__(self, topology, dimension=2):
    self.name ='Rastrigin'
    self.topology = topology
    self.domain = (-5.12,5.12)
    self.dimension = dimension
    self.optimalValue = 0
    self.optimalPosition = np.zeros((self.dimension,))
  def eval(self, X, **kwargs):
    A = kwargs.get('A', 10)
    # return A*len(X) + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])
    return A*len(X) + sum(X**2 - A * np.cos(2 * math.pi * X))
  
  def distanceOptimal(self, position):
    return np.linalg.norm(self.optimalPosition - position)
    # return np.abs(compareValue - self.optimalValue)

### F2: Rosenbrock
class Rosenbrock():
  def __init__(self, topology, dimension=2):
    self.name ='Rosenbrock'
    self.topology = topology
    self.domain = (-EXTREME,EXTREME)
    self.dimension = dimension
    self.optimalValue = 0
    self.optimalPosition = np.ones((self.dimension,))
  def eval(self, X, **kwargs):
    return sum(100*(X[1:]- X[:-1]**2)**2 + (1 - X[:-1])**2)
  
  def distanceOptimal(self, position):
    # print(np.linalg.norm(self.optimalPosition - position))
    # aaa
    return np.linalg.norm(self.optimalPosition - position)
    # return np.abs(compareValue - self.optimalValue)

### F3: Eggholder
class Eggholder():
  def __init__(self, topology, dimension=2):
    if (dimension != 2):
      raise Exception('Dimension must be 2')
    self.name ='Eggholder'
    self.topology = topology
    self.domain = (-512,512)
    self.dimension = dimension
    self.optimalValue = -959.6407
    self.optimalPosition = (512, 404.2319)
  def eval(self, X, **kwargs):
    x = X[0]
    y = X[1]
    return -(y+47)*np.sin(np.sqrt(np.abs(x/2+(y+47)))) - x*np.sin(np.sqrt(np.abs(x-(y+47))))
  
  def distanceOptimal(self, position):
    return np.linalg.norm(self.optimalPosition - position)
    # return np.abs(compareValue - self.optimalValue)

### F4: Ackley
class Ackley():
  def __init__(self, topology, dimension=2):
    if (dimension != 2):
      raise Exception('Dimension must be 2')
    self.name ='Ackley'
    self.topology = topology
    self.domain = (-5,5)
    self.dimension = dimension
    self.optimalValue = 0
    self.optimalPosition = (0, 0)
  def eval(self, X, **kwargs):
    x = X[0]
    y = X[1]
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) -np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.e + 20
  
  def distanceOptimal(self, position):
      return np.linalg.norm(self.optimalPosition - position)
    # return np.abs(compareValue - self.optimalValue)

class Particle():
  def __init__(self, func, index, neighborsIndex,paramConfig):
    self.index = index
    self.domain = func.domain
    self.dimension = func.dimension
    initPosition = (self.domain[1] - self.domain[0]) * np.random.random_sample(self.dimension) + self.domain[0]
    initState = {
        'position': initPosition,
        'value': func.eval(initPosition)
    }
    self.paramConfig = paramConfig
    self.best = initState
    self.current = initState.copy()
    self.neighborsIndex = neighborsIndex
    self.inertiaComponent = np.zeros(self.dimension,)
    self.velocity = np.zeros(self.dimension,)
    velocityMax = self.paramConfig.get('k') *(self.domain[1] - self.domain[0]) / 2
    self.velocityRange = (-velocityMax, velocityMax)
    self.func = func
  def updateSocial(self, bestPositionNeighbor):
    self.socialComponent = (bestPositionNeighbor - self.current["position"]) * self.paramConfig.get('c2') * np.random.random_sample(self.dimension)
  def updateCognitive(self):
    self.cognitiveComponent = (self.best["position"] - self.current["position"]) * self.paramConfig.get('c1') * np.random.random_sample(self.dimension)
  def updateVelocity(self):
    self.velocity = self.velocity * self.paramConfig.get('w') + self.cognitiveComponent + self.socialComponent
    for i,d in enumerate(self.velocity):
      if (d > self.velocityRange[1]):
        self.velocity[i] = self.velocityRange[1]
      elif (d < self.velocityRange[0]):
        self.velocity[i] = self.velocityRange[0]
  def updateState(self):
    newPosition = self.velocity + self.current["position"]
    for i,d in enumerate(newPosition):
        if (d > self.func.domain[1]):
            newPosition[i] = self.func.domain[1]
        elif (d < self.func.domain[0]):
            newPosition[i] = self.func.domain[0]
    newValue = self.func.eval(newPosition)
    self.current = {
        'position': newPosition,
        'value': newValue
    }
    if (newValue <= self.best['value']):
      self.best = copy.deepcopy(self.current)


class PSO():
  def __init__(self,func,paramConfig,maxEvaluations=None,N=32,generation=50):
    self.N = N
    self.func = func
    self.topology = self.func.topology
    self.generation = generation
    self.numEvaluations = 0
    self.maxEvaluations = maxEvaluations
    # Step1: Initial swarm
    particles = [] ## len == N
    for index in range(N):
      neighborsIndex = None
      if (self.topology == 'ring'):
        if (index == N - 1):
          neighborsIndex = [index - 1, index, 0]
        elif (index == 0):
          neighborsIndex = [N-1, index, 1]
        else:
          neighborsIndex = [index-1, index, index+1]
      particle = Particle(func, index, neighborsIndex, paramConfig)
      particles.append(particle)
    bestParticleIndex = np.argmin([particle.current["value"] for particle in particles])
    bestParticle = particles[bestParticleIndex]
    self.numEvaluations += self.N
    self.particles = particles
    self.bestParticle = bestParticle
    self.bestParticleList = []
    self.currentGeneration = 0
    self.posParticleList = []
    self.avgFitnessList = []
    self.updateListSoFar()
  def step(self):
    # If run max evaluations
    if (self.maxEvaluations and self.numEvaluations + self.N > self.maxEvaluations):
      print(f'Stop at generation: {self.currentGeneration}, Total evaluations: {self.numEvaluations}')
      return None
    # If run all generation
    if (self.maxEvaluations == None and self.currentGeneration == self.generation):
      print('Already run {} generations'.format(self.generation))
      return None
    # Else
    for particle in self.particles:
      # Step2: Update social components
      if self.topology == 'ring':
        neighbors = [self.particles[index] for index in particle.neighborsIndex]
        # print(particle.neighborsIndex)
        bestNeighborIndex = np.argmin([x.best["value"] for x in neighbors])
        particle.updateSocial(neighbors[bestNeighborIndex].best["position"])
      elif (self.topology == 'star'):
        bestNeighborIndex = np.argmin([particle.best["value"] for particle in self.particles])
        bestNeighbor = self.particles[bestNeighborIndex]
        particle.updateSocial(bestNeighbor.best["position"])
      # Step3: Update State
      particle.updateCognitive()
      particle.updateVelocity()
      particle.updateState()
      if (particle.current['value'] < self.bestParticle.current['value']):
        # print(particle.current['value'])
        self.bestParticle = copy.deepcopy(particle)
    self.currentGeneration += 1
    self.numEvaluations += self.N
    self.updateListSoFar()
    return True

  def updateListSoFar(self):
    self.bestParticleList.append(self.bestParticle)
    self.posParticleList.append([particle.current['position'] for particle in self.particles])
    self.avgFitnessList.append(np.mean([particle.current['value'] for particle in self.particles]))

  def getAllCurrentPositions(self):
    return np.array([particle.current['position'] for particle in self.particles])

  def runAll(self):
    while (True):
        if (self.step() == None): break

  def getReport(self, seed):
    bestFinessValue = self.bestParticle.current['value']
    reportDict = {
        'SEED': f'{seed}',
        'Generation: ': self.currentGeneration,
        'Number of evaluations: ': self.numEvaluations,
        'Best position: ': self.bestParticle.current['position'],
        'Best finess value: ': bestFinessValue,
        'Distance to minimum: ': self.func.distanceOptimal(self.bestParticle.current['position'])
    }    
    return reportDict

  def visualize(self, savePath, gridSize=50, fps=4):
    def init():
      ax1.set_xlim([-self.func.domain[0], -self.func.domain[1]])
      ax1.set_ylim([-self.func.domain[0], -self.func.domain[1]])
      ax2.set_xlabel('Generations')
      ax2.set_ylabel('Average fitness')

    def animate(i):
      particlePoints.set_offsets(self.posParticleList[i])
      ax1.set_title((f'Generation {i}'))
      ax2.plot(np.arange(i+1),self.avgFitnessList[:i+1],c='g')
      ax2.set_title('Best value: {:0.5f}'.format(self.bestParticleList[i].current['value']))
    fig, (ax2, ax1) = plt.subplots(1, 2)
    # Setup config
    X = np.linspace(-self.func.domain[0], -self.func.domain[1], gridSize)
    Y = np.linspace(-self.func.domain[0], -self.func.domain[1], gridSize)
    X, Y = np.meshgrid(X, Y)
    # print(len([X,Y]))
    Z = self.func.eval(np.array([X, Y]))
    cp =  ax1.contourf(X, Y, Z, zorder=0)
    fig.colorbar(cp)
    fig.set_size_inches(10, 5)
    fig.suptitle("{}: {} topology".format(self.func.name, self.func.topology))
    
    particlePoints = ax1.scatter([],[], c='r',zorder=1)
    
    ani = FuncAnimation(fig, animate, np.arange(self.currentGeneration), init_func=init)
    writer = PillowWriter(fps=fps)  
    ani.save(f"{savePath}/{self.func.name.lower()}_{self.func.topology}_{fps}fps.gif", writer=writer, dpi=100)   
    plt.show()
