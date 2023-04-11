import numpy as np
import struct
import torch
import random
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


def blendGenome(gamete1, gamete2):
    parent1 = gamete1.split()
    parent2 = gamete2.split()
    newGenome = ""
    for chromosome in range(28):
        if chromosome % 2 == 0:
            newGenome += parent1[chromosome] + " "
        else:
            newGenome += parent2[chromosome] + " "
    return newGenome[:-1]


class Agent:
    def __init__(self, _loc, gamete1, gamete2):
        self.loc = _loc
        self.gamete1 = gamete1
        self.gamete2 = gamete2
        self.genome = blendGenome(gamete1, gamete2)
        weights = self.genome.split()
        weights = list(map(hexToFloat, weights))
        self.brain = NeuralNet(weights)
        self.alive = True

    def getStats(self):
        print(self.loc)
        print(self.genome)

    def makeDecision(self, inputArray, obst):
        inputT = torch.tensor(inputArray)
        outputT = self.brain(inputT)
        dec = torch.argmax(outputT).item()
        verbose = ["up", "down", "left", "right"]
        if dec in get_movement_choices(self.loc, (dim, dim), obst):
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

    def getGamete(self):
        parent1 = self.gamete1.split()
        parent2 = self.gamete2.split()
        gamete = ""
        for chromosome in range(28):
            randInt = np.random.randint(0, 17)
            gamete += (
                parent1[chromosome][:randInt] + parent2[chromosome][randInt:] + " "
            )
        return gamete[:-1]


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


def get_movement_choices(loc, constraints, obstacles=None):
    choices = [0, 1, 2, 3]
    if loc[1] == 0 or (loc[0], loc[1] - 1) in obstacles:  # N
        choices.pop(choices.index(0))
    if loc[0] == constraints[0] or (loc[0] + 1, loc[1]) in obstacles:  # E
        choices.pop(choices.index(3))
    if loc[1] == constraints[1] or (loc[0], loc[1] + 1) in obstacles:  # S
        choices.pop(choices.index(1))
    if loc[0] == 0 or (loc[0] - 1, loc[1]) in obstacles:  # W
        choices.pop(choices.index(2))
    return choices


def rect_obst(tl, w, h):
    coords = []
    for i in range(tl[0], tl[0] + w):
        for j in range(tl[1], tl[1] + h):
            coords.append((i, j))
    return coords


def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


def rand_not_obst(low, high, obst):
    while True:
        coord = (random.randint(low, high), random.randint(low, high))
        if coord not in obst:
            return coord


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

G = 50  # number of generations
t = 100  # timesteps per generation
N = 40  # initial number of creatures
dim = 100
peeps = []  # list of creatures
options = [1.0, 2.0, 3.0]
record = True  # Controls whether or not to store gifs of run
survivalRate = []
genBucket = []

obst = rect_obst((60, 15), 10, 30)

for creature in range(N):
    peeps.append(
        Agent(
            rand_not_obst(0, dim, obst),
            randomGenome(),
            randomGenome(),
        )
    )

survival_percentages = []

for generation in range(G):
    if generation % 5 == 0:
        record = True
    else:
        record = False

    # Random Placement
    for creature in range(len(peeps)):
        peeps[creature].loc = rand_not_obst(0, dim, obst)

    # Movement step
    for step in range(t):
        for creature in range(len(peeps)):
            decision = peeps[creature].makeDecision(
                [peeps[creature].loc[0], peeps[creature].loc[1], 0.0], obst
            )
            locs = [peep.loc for peep in peeps]
            x = [x[0] for x in locs]
            y = [y[1] for y in locs]
        if record:
            x_obst = [x[0] for x in obst]
            y_obst = [y[1] for y in obst]
            plt.scatter(x_obst, y_obst)
            plt.scatter(x, y)
            plt.xlim(0, dim)
            plt.ylim(0, dim)
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
        if peeps[creature].loc[0] < dim / 2:
            peeps[creature].alive = False
    peeps = [peep if peep.alive else None for peep in peeps]
    peeps = list(filter(lambda item: item is not None, peeps))
    survivalRate.append(len(peeps) / N)
    genBucket.append(generation + 1)

    peeps_c = peeps.copy()
    # Reproduction step

    pairs = []
    while len(peeps_c) > 1:
        rand1 = pop_random(peeps_c)
        rand2 = pop_random(peeps_c)
        pair = rand1, rand2
        pairs.append(pair)

    peeps = []
    for newPeep in range(N):
        pair = pairs[random.randint(0, len(pairs) - 1)]
        parent1 = pair[0].getGamete()
        parent2 = pair[1].getGamete()
        peeps.append(Agent((rand_not_obst(0, dim, obst)), parent1, parent2))


# for generation
#   for timestep
#       run test
#       get surviving agents
#       Simulate Mating
#           Group by trait value to emulate assortative mating
#           Take half of each genome randomly and combine them
#           Make sure that reproduction is done in the excess of resources


# print("Init time: {}, Dec time: {}".format(endInit - startInit, endDec - startDec))
plt.plot(genBucket, survivalRate)
plt.title("Survival Rate Across Generations")
plt.xlabel("Generation")
plt.ylabel("Survival Rate")
plt.show()
