import random
import pygame
import math
import numpy as np
import sys
import string
import time
#from nn import NeuralNetwork
from nn2 import NeuralNetwork

results_filename = "results_" + str(time.time()) + ".csv"
fp = open(results_filename, 'a')
fp.write("generation,total,avg,max\n")

(width, height) = (800, 800)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Insects simulation')

lines = [
        [20, int(height/2)],
        [100, 200],
        [300, 200],
        [350, 400]
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
        self.x = lines[0][0] + random.randint(0, 20) - 10
        self.y = lines[0][1] + random.randint(0, 20) - 10
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.orientation = 2 * math.pi * random.random()
        self.size = 10
        self.health = 500
        self.cntrLived = 0
        self.isAlive = True
        self.fitness = 1
        self.brain = NeuralNetwork(1, 1, 4)
        self.currentTargetIdx = 1

    def think(self, target):
        target_orientation = math.atan2(target['point']['y'] - self.y, target['point']['x'] - self.x)
        target_orientation += math.pi 
        target_orientation = renormalize(target_orientation, [0,  2 * math.pi], [0, 1])
        distance = renormalize(target['distance'], [0, 1131], [0, 1])
        inp_vec = [target_orientation]
        result = self.brain.run(inp_vec)
#        print('inp', inp_vec, 'result', result)
#        gas = result[0]
        gas = 5 # constant gas for now
        steer = result[0]
#        gas = renormalize(gas, [0, 1], [0, 15])
        steer = renormalize(steer, [0, 1], [-15, 15]) 
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
            if desired_target['distance'] < self.size:
                print("HIT!")
                self.health += 40
                self.fitness += 1
                self.currentTargetIdx += 1

        (gas, steer) = self.think(desired_target)
#        print("GAS: {} STEER: {}".format(gas, steer))
#        gas = self.gas()
#        steer = self.steer()
        if gas <= 3:
            return

        pseudo_target_x = self.x + int(math.cos(self.orientation) * gas)
        pseudo_target_y = self.y + int(math.sin(self.orientation) * gas)
#        print("{}, {} (gas: {}, steer: {}, ori: {}) ----> {}, {}".format(self.x, self.y, gas, steer, self.orientation, pseudo_target_x, pseudo_target_y))

        # steer is based on perpendicular vector with controlled length
        st_dir = self.orientation + math.pi / 2
        target_x = pseudo_target_x + int(math.cos(st_dir) * steer)
        target_y = pseudo_target_y + int(math.sin(st_dir) * steer)

        # steer is based on controlled angle
#        target_x = self.x + int(math.cos(steer) * 5)
#        target_y = self.y + int(math.sin(steer) * 5)


        self.orientation = math.atan2(target_y - self.y, target_x - self.x)
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
num_insects = 50
mating_pool = []
max_mating_pool = 50
mutating_chance = 1 # %
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

    # evaluate fitness phase
    if numAlive == 0:
        (total_fitness, max_fitness, avg_fitness, total_mating_slots) = (0, 1, 0, 0)
        for i in insects:
            total_fitness += i.fitness
            if i.fitness >= max_fitness:
                max_fitness = i.fitness
        avg_fitness = int(total_fitness / len(insects))
        print("Generation: {} fitness total={} avg={} max={}".format(generation_cntr, total_fitness, avg_fitness, max_fitness))
        fp = open(results_filename, 'a')
        fp.write("{},{},{},{}\n".format(generation_cntr, total_fitness, avg_fitness, max_fitness))
        fp.close()

        for i in insects:
            i.desired_mating_slots = i.fitness 
            # above average score is rewarded with extra mating slots
            if i.fitness > avg_fitness:
                i.extra_mating_slots = (i.fitness - avg_fitness) ** 2
                i.desired_mating_slots += i.extra_mating_slots
            total_mating_slots += i.desired_mating_slots
            
        # mating phase
        mating_pool = []
        for i in insects:
            mating_slots = round(max_mating_pool * i.desired_mating_slots / total_mating_slots)
            print("Individual {} ({}/{}) fed: {} lived: {}, desired mating slots: {} actual: {}".format(
                i.id, i.parentAid, i.parentBid, i.fitness, i.cntrLived, i.desired_mating_slots, mating_slots
                ))
            for c in range(mating_slots):
                mating_pool.append(i)

        print("Mating pool size: {}".format(len(mating_pool)))
        next_generation = []
        for _ in range(num_insects):
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            kid = parent1.mate_with(parent2)
            kid.mutate()
            next_generation.append(kid)

        insects = next_generation
        generation_cntr += 1
        

    if not isRunningFast:
        pygame.time.delay(20)


    pygame.draw.lines(screen, (0, 0, 255), False, lines, 3)
    pygame.display.flip()
