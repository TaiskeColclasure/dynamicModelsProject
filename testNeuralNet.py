import numpy as np
import struct
import torch
import time
import random
from itertools import combinations
import math
import matplotlib.pyplot as plt
import imageio
import os


class NeuralNet(torch.nn.Module):
    def __init__(self, weights):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        inputLayer = torch.tensor(
            [weights[:3], weights[3:6], weights[6:9], weights[9:12]]
        )
        outputLayer = torch.tensor(
            [weights[12:16], weights[16:20], weights[20:24], weights[24:]]
        )
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
        self.alive = True

    def getStats(self):
        print(self.loc)
        print(self.genome)

    def makeDecision(self, inputArray):
        inputT = torch.tensor(inputArray)
        outputT = self.brain(inputT)
        dec = torch.argmax(outputT).item()
        verbose = ["up", "down", "left", "right"]
        if dec in get_movement_choices(self.loc, (16, 16)):
            if dec == 2:
                self.loc = (self.loc[0] - 1, self.loc[1])
            if dec == 3:
                self.loc = (self.loc[0] + 1, self.loc[1])
            if dec == 0:
                self.loc = (self.loc[0], self.loc[1] - 1)
            if dec == 1:
                self.loc = (self.loc[0], self.loc[1] + 1)
            return (dec, verbose[dec])
        else:
            return None


# Helper function to create hex value using our custom convention
def floatToHex(_float):
    unsigned = hex(struct.unpack("<Q", struct.pack("<d", _float))[0])
    if _float < 0:
        return "1" + unsigned[1:]
    else:
        return unsigned


# Helper function to convert number types using convention where digit is negative if the first char is 1
def hexToFloat(_hex):
    signed = struct.unpack(">d", bytes.fromhex(_hex[2:]))[0]
    return signed


# This function generates 31 random hex values to initialize a random genome to start the population with
def randomGenome():
    _genome = ""
    data = np.random.normal(0, 1 / 3, 28)
    data = np.clip(data, -1, 1)
    for chromosome in data:
        randomHex = floatToHex(chromosome)
        _genome += randomHex + " "
    return _genome[:-1]


def distance(p1, p2):
    d1 = p2[0] - p1[0]
    d2 = p2[1] - p1[1]
    return math.sqrt(d1**2 + d2**2)


def get_movement_choices(loc, constraints):
    choices = [0, 1, 2, 3]
    if loc[1] == 0:  # N
        choices.pop(choices.index(0))
    if loc[0] == constraints[0]:  # E
        choices.pop(choices.index(3))
    if loc[1] == constraints[1]:  # S
        choices.pop(choices.index(1))
    if loc[0] == 0:  # W
        choices.pop(choices.index(2))
    return choices


def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


# def move(action, loc):
#     if action[1] == "left":
#         return (loc[0] - 1, loc[1])
#     if action[1] == "right":
#         return (loc[0] + 1, loc[1])
#     if action[1] == "up":
#         return (loc[0], loc[1] - 1)
#     if action[1] == "down":
#         return (loc[0], loc[1] + 1)


# startInit = time.time()
# test = Agent((1, 1), randomGenome())
# endInit = time.time()

# startDec = time.time()
# print(test.getDecision([1.0, 2.0, 3.0]))
# endDec = time.time()

G = 5  # number of generations
t = 50  # timesteps per generation
N = 16  # initial number of creatures
peeps = []  # list of creatures
options = [1.0, 2.0, 3.0]
record = True  # Controls whether or not to store gifs of run

for creature in range(N):
    peeps.append(Agent((random.randint(0, 16), random.randint(0, 16)), randomGenome()))

survival_percentages = []

for generation in range(G):
    # Random Placement
    for creature in range(len(peeps)):
        peeps[creature].loc = (random.randint(0, 16), random.randint(0, 16))

    # Movement step
    for step in range(t):
        for creature in range(len(peeps)):
            decision = peeps[creature].makeDecision(
                [peeps[creature].loc[0], peeps[creature].loc[1], 0.0]
            )
            locs = [peep.loc for peep in peeps]
            x = [x[0] for x in locs]
            y = [y[1] for y in locs]
            if record:
                plt.scatter(x, y)
                plt.xlim(0, 16)
                plt.ylim(0, 16)
                plt.savefig("frames/bruh{}.png".format(step))
                # plt.show()
                plt.close()
    if record:
        frames = []
        for i in range(t):
            image = imageio.v2.imread("frames/bruh{}.png".format(i))
            frames.append(image)
        imageio.mimsave("gen{}.gif".format(generation), frames)
        os.popen("rm frames/bruh*")

    # Evaluation step
    for creature in range(len(peeps)):
        if peeps[creature].loc[0] < 8:
            peeps[creature].alive = False
    peeps = [peep if peep.alive else None for peep in peeps]
    peeps = list(filter(lambda item: item is not None, peeps))

    peeps_c = peeps.copy()
    # Reproduction step

    pairs = []
    while len(peeps_c) > 1:
        rand1 = pop_random(peeps_c)
        rand2 = pop_random(peeps_c)
        pair = rand1, rand2
        pairs.append(pair)

    for newPeep in range(N):
        pair = pairs[random.randint(0, len(pairs) - 1)]
        parent1 = pair[0].genome.split()
        parent2 = pair[1].genome.split()
        newGenome = ""
        for chromosome in range(28):
            if chromosome % 2 == 0:
                newGenome += parent1[chromosome] + " "
            else:
                newGenome += parent2[chromosome] + " "
        newGenome = newGenome[:-1]
        peeps.append(Agent((0, 0), newGenome))
    
    curr_survival_percentage = len(peeps) / N
    survival_percentages.append(curr_survival_percentage)

    plt.plot(survival_percentages)
    plt.title('Survival Rate Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Survival Rate')
    plt.show()

# for generation
#   for timestep
#       run test
#       get surviving agents
#       Simulate Mating
#           Group by trait value to emulate assortative mating
#           Take half of each genome randomly and combine them
#           Make sure that reproduction is done in the excess of resources


# print("Init time: {}, Dec time: {}".format(endInit - startInit, endDec - startDec))
