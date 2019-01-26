import random
import pygame
import math
import numpy as np
import sys
import string
import time
#from nn import NeuralNetwork
from nn2 import NeuralNetwork

results_filename = "logs/results_" + str(time.time()) + ".csv"
fp = open(results_filename, 'a')
fp.write("generation,total,avg,max\n")

(width, height) = (800, 800)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Insects simulation')

lines = [
        [220, int(height/2)],
        [250, 280],
        [350, 220],
        [500, 200],
        [550, 400],
        [620, 440],
        [650, 500],
        [670, 520],
        [700, 600],
        ]

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def randomid(n = 3):
    out = []
    for _ in range(n):
        out.append(random.choice(string.ascii_uppercase))
    return ''.join(out)

class Insect:
    def __init__(self):
        self.id = randomid()
        self.parentAid = ''
        self.parentBid = ''
        # random starting point within 10px of lines[0]
#        self.x = lines[0][0] + random.randint(0, 20) - 10
#        self.y = lines[0][1] + random.randint(0, 20) - 10
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.orientation = 2 * math.pi * random.random()
        self.size = 10
        self.health = 200
        self.cntrLived = 0
        self.isAlive = True
        self.fitness = 0
        self.brain = NeuralNetwork(1, 1, 4)
        self.currentTargetIdx = 0
        self.desiredMatingSlots = 0
        self.minDistances = []

    def think(self, target):
        target_orientation = math.atan2(target['point']['y'] - self.y, target['point']['x'] - self.x)
#        target_orientation += math.pi 
        target_orientation = renormalize(target_orientation, [- math.pi,  math.pi], [0, 1])
        inp_vec = [target_orientation]
        result = self.brain.run(inp_vec)
#        print('inp', inp_vec, 'result', result)
#        gas = result[0]
        gas = 5 # constant gas for now
        steer = result[0]
#        gas = renormalize(gas, [0, 1], [0, 15])
        steer = renormalize(steer, [0, 1], [- math.pi, math.pi]) 
#        print("IN: {} OUT: {}".format(target_orientation, steer))
        return (gas, steer)

    def update(self):
        self.health -= 1
        if self.health <= 0:
            self.isAlive = False

        if not self.isAlive:
            return

        desired_target = self.find_closest_target()
        if desired_target['point']:
            pygame.draw.aaline(screen, (255, 0, 0), (self.x, self.y), (desired_target['point']['x'], desired_target['point']['y']))
            if len(self.minDistances) < self.currentTargetIdx + 1:
                self.minDistances.append(int(desired_target['distance']))
            if desired_target['distance'] < self.minDistances[self.currentTargetIdx]:
                desired_target['distance'] = self.minDistances[self.currentTargetIdx]
            if desired_target['distance'] < self.size:
                self.health += 40
                self.currentTargetIdx += 1

        (gas, steer) = self.think(desired_target)
#        print("GAS: {} STEER: {}".format(gas, steer))
#        gas = self.gas()
#        steer = self.steer()
        if gas <= 3:
            return

        self.orientation = steer
        target_x = self.x + int(math.cos(self.orientation) * gas)
        target_y = self.y + int(math.sin(self.orientation) * gas)
#        print("{}, {} (gas: {}, steer: {}, ori: {}) ----> {}, {}".format(self.x, self.y, gas, steer, self.orientation, pseudo_target_x, pseudo_target_y))

        self.x = target_x
        self.y = target_y
        self.cntrLived += 1

    def display(self):
        if not self.isAlive:
            return
        # draw body
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size, 2)
        # draw antennas
        antena_len = 20
        antena_start = (self.x, self.y)
        antena_end = (
                antena_start[0] + math.cos(self.orientation) * antena_len,
                antena_start[1] + math.sin(self.orientation) * antena_len
                )
        antena = pygame.draw.aaline(screen, self.color, antena_start, antena_end)

    # go trough all line segments
    # find closest distance
    def find_closest_target(self):
        min_distance = None 
        selected_target = None
        if self.currentTargetIdx < len(lines):
            target = lines[self.currentTargetIdx]
            selected_target = {'x': target[0], 'y': target[1]}
            min_distance = math.hypot(target[0] - self.x, target[1] - self.y)
            
        return {'point': selected_target, 'distance': min_distance}
    
    def mate_with(self, parent):
        kid = Insect()
        kid.parentAid = self.id
        kid.parentBid = parent.id

        selfLinearW1 = [item for sublist in self.brain.W1 for item in sublist] #np.concatenate(self.brain.W1)
        parentLinearW1 = [item for sublist in parent.brain.W1 for item in sublist] #np.concatenate(parent.brain.W1)
        median = random.randint(0, len(selfLinearW1))
        kidLinearW1 = np.concatenate([selfLinearW1[0:median], parentLinearW1[median:]])
        kid.brain.W1 = np.split(kidLinearW1, 1)

        selfLinearW2 = [item for sublist in self.brain.W2 for item in sublist] #np.concatenate(self.brain.W2)
        parentLinearW2 = [item for sublist in parent.brain.W2 for item in sublist] #np.concatenate(parent.brain.W2)
        median = random.randint(0, len(selfLinearW2))
        kidLinearW2 = np.concatenate([selfLinearW2[0:median], parentLinearW2[median:]])
        kid.brain.W2 = np.split(kidLinearW2, 4)
        return kid

    def mutate(self):
        if random.randint(0, 100) < mutating_chance:
            if random.randint(0, 1) == 1:
                # mutate W1
                i = random.randint(0, 0)
                j = random.randint(0, 3)
                self.brain.W1[i][j] = random.random()
            else:
                # mutate W2
                i = random.randint(0, 3)
                j = random.randint(0, 0)
                self.brain.W2[i][j] = random.random()


insects = []
num_insects = 70
mating_pool = []
max_mating_pool = 70
mutating_chance = 5 # %
generation_cntr = 1

for _ in range(num_insects):
    insect = Insect()
    insects.append(insect)


running = True
isRunningFast = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print
            if pygame.key.get_pressed()[pygame.K_s]:
                isRunningFast ^= 1

        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    # population develop phase
    numAlive = 0
    for i in insects:
        i.update()
        i.display()
        if i.isAlive:
            numAlive += 1

    # end current generation
    if numAlive == 0:
        # evaluate fitness phase
        (total_fitness, max_fitness, avg_fitness, total_mating_slots) = (0, 1, 0, 0)

        # simplify min distances array
        min_distances = [None] * 99
        max_distances = [None] * 99
        for i in insects:
            for j in range(len(lines)):
                if len(i.minDistances) >= j + 1 and min_distances[j] == None:
                    min_distances[j] = i.minDistances[j]
                if len(i.minDistances) >= j + 1 and min_distances[j] > i.minDistances[j]:
                    min_distances[j] = i.minDistances[j]
                if len(i.minDistances) >= j + 1 and max_distances[j] == None:
                    max_distances[j] = i.minDistances[j]
                if len(i.minDistances) >= j + 1 and max_distances[j] < i.minDistances[j]:
                    max_distances[j] = i.minDistances[j]

        print("Min distances: ", min_distances)
        print("Max distances", max_distances)

        for i in insects:
            i.fitness = 1
            # calculate fitness for every attemptet target
            for idx, minval in enumerate(i.minDistances):
                renormalized = int(renormalize(minval, (max_distances[idx]+1, min_distances[idx]-1), (0, 10)))
                thisfitness = renormalized * 10 ** idx
                # print("Renormalized", minval,"=>",renormalized, "fit", thisfitness)
                i.fitness += thisfitness

            total_fitness += i.fitness
            if i.fitness >= max_fitness:
                max_fitness = i.fitness
        avg_fitness = int(total_fitness / len(insects))
        fp = open(results_filename, 'a')
        fp.write("{},{},{},{}\n".format(generation_cntr, total_fitness, avg_fitness, max_fitness))
        fp.close()

        for i in insects:
            if i.fitness >= 3:
                i.desiredMatingSlots = i.fitness - 1 
                # above average score is rewarded with extra mating slots
                if i.fitness > avg_fitness:
                    i.extra_mating_slots = (i.fitness - avg_fitness) ** 2
                    i.desiredMatingSlots += i.extra_mating_slots
            total_mating_slots += i.desiredMatingSlots
            
        # mating phase
        mating_pool = []
        if total_mating_slots > 0:
            for i in insects:
                mating_slots = round(max_mating_pool * i.desiredMatingSlots / total_mating_slots)
                print("Individual {} ({}/{}) fitness: {} targetIdx: {}, desired mating slots: {} actual: {}".format(
                    i.id, i.parentAid, i.parentBid, i.fitness, i.currentTargetIdx, i.desiredMatingSlots, mating_slots
                    ))

                for c in range(mating_slots):
                    mating_pool.append(i)
        print("Generation: {} fitness total={} avg={} max={}".format(generation_cntr, total_fitness, avg_fitness, max_fitness))
        print("Mating pool size: {}/{}".format(len(mating_pool), max_mating_pool))

        # add blank mates if previous generation was useless
        while len(mating_pool) < max_mating_pool:
            mating_pool.append(Insect())

        next_generation = []
        for _ in range(num_insects):
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            kid = parent1.mate_with(parent2)
            kid.mutate()
            next_generation.append(kid)

        # add some blanks for the sake of randomness
        for _ in range(5):
            next_generation.append(Insect())
        insects = next_generation
        generation_cntr += 1
        

    if not isRunningFast:
        pygame.time.delay(20)


    pygame.draw.lines(screen, (0, 0, 255), False, lines, 3)
    pygame.display.flip()
