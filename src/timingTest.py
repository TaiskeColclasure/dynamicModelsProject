import numpy as np
import struct
import torch
import time

class NeuralNet(torch.nn.Module):
    def __init__(self, weights):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(3,4)
        self.fc2 = torch.nn.Linear(4,4)
        inputLayer = torch.tensor([
            weights[:3],
            weights[3:6],
            weights[6:9],
            weights[9:12]
        ])
        outputLayer = torch.tensor([
            weights[12:16],
            weights[16:20],
            weights[20:24],
            weights[24:]
        ])
        self.relu = torch.nn.ReLU()
        self.fc1.weight.data = inputLayer
        self.fc2.weight.data = outputLayer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, _loc, _genome):
        self.loc = _loc
        self.genome = _genome
        weights = _genome.split()
        weights = list(map(hexToFloat, weights))
        self.brain = NeuralNet(weights)
        
    def getStats(self):
        print(self.loc)
        print(self.genome)

    def getDecision(self, inputArray):
        inputT = torch.tensor(inputArray)
        outputT = self.brain(inputT)
        dec = torch.argmax(outputT).item()
        verbose = ['up', 'down', 'left', 'right']
        return (dec, verbose[dec])

class World:
    def __init__(self, dimTuple, n, genMax, lifeCycle):
        self.N = n
        self.dim = dimTuple
        self.genMax = genMax
        self.lifeCycle = lifeCycle
        self.population = []
        for i in range(n):
            floatArray = np.random.uniform(-100,100, 2)
            self.population.append(Agent((int(floatArray[0]), int(floatArray[1])), randomGenome()))


    def runSim(self):
        for generation in range(self.genMax):
            for t in range(self.lifeCycle):
                for agent in self.population:



# Helper function to create hex value using our custom convention
def floatToHex(_float):
    unsigned = hex(struct.unpack('<Q', struct.pack('<d', _float))[0])
    if (_float < 0):
        return '1' + unsigned[1:]
    else:
        return unsigned
# Helper function to convert number types using convention where digit is negative if the first char is 1
def hexToFloat(_hex):
    signed = struct.unpack('>d', bytes.fromhex(_hex[2:]))[0]
    return signed
# This function generates 31 random hex values to initialize a random genome to start the population with
def randomGenome():
    _genome = ""
    data = np.random.normal(0, 1/3, 28)
    data = np.clip(data, -1, 1)
    for chromosome in data:
        randomHex = floatToHex(chromosome)
        _genome += randomHex + ' '
    return _genome[:-1]

agents = []
for i in range(200):
    agents.append(Agent((0,0), randomGenome()))


startInit = time.time()
test = Agent((1,1), randomGenome())
endInit = time.time()

startDec = time.time()
for t in range(120):
    for entity in agents:
        entity.getDecision([1.0,2.0,3.0])
endDec = time.time()

print('Init time: {}, Dec time: {}'.format(endInit - startInit, endDec - startDec))
world = World((200,200), 1)

for newPeep in range(populationSize):
    pair = np.random.choice(peep_c, replace=False)
    parent1 = pair[0].genome.split()
    parent2 = pair[1].genome.split()
    newGenome = ''
    for chromosome in range(28):
        if chromosome % 2 == 0:
            newGenome += parent1[chromosome] + ' '
        else:
            newGenome += parent2[chromosome] + ' '
    newGenome = newGenome[:-1]

