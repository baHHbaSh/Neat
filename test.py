import traceback
import neat
import os
import gc
import math
import keyboard
from turtle import*
from random import randint

class player(Turtle):
    def __init__(self):
        super().__init__()
        self.PlayerRot = 0
        self.color(randint(0, 255)/255,randint(0, 255)/255,randint(0, 255)/255)
        self.speed = 9999999
        self.Rate = 0
        self.Iter = 0
        self.DefaultDistance = self.Distance()
        self.penup()
        self.CanMove = True
    def Distance(self):
        global Target
        return math.sqrt((Target.position()[0] - self.position()[0]) ** 2 + (Target.position()[1] - self.position()[1]) ** 2)
    def Log(self):
        return [self.position()[0], self.position()[1], self.Distance(), self.PlayerRot % 360]
    def Move(self, speed):
        if self.CanMove:
            self.forward(speed * 3)
            self.Iter+=1
    def Rotate(self, Angle):
        if self.CanMove:
            self.PlayerRot += Angle*5
            try:
                self.setheading(self.PlayerRot)
            except: print(traceback.format_exc())
    def SelfRate(self):
        return self.DefaultDistance / self.Distance() - self.Iter / 500
    def OnWin(self):
        return True if self.Distance() < 2 else False

Target = Turtle()
Target.penup()
Target.goto(randint(-400, 400), randint(-400, 400))

config = neat.Config(neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "data.txt")

generation = 0

Tutels = []

def StartLearn(genoms, config):
    Target.goto(randint(-200, 200), randint(-200, 200))

    global Tutels, Check
    nets = []
    Players = []


    for id, g in genoms:
        nets.append(
            neat.nn.FeedForwardNetwork.create(g, config)
        )
        g.fitness = 0

        Players.append(player())


    global generation
    generation += 1
    print("Gen =", generation)

    print(len(nets), len(Players))
    Learn = True
    while Learn:
        if len(Players) <= 1: break

        for index, Player in enumerate(Players):
            output = nets[index].activate(Player.Log())#[-1 <= x <=1, ...]
            Player.Move(output[0])
            Player.Rotate(output[1])
            Player.Rotate(-output[2])
            #Rate
            Player.Rate += 10/Player.Log()[2] - .1
            genoms[index][1].fitness = Player.SelfRate()

            Winner = Player.OnWin()
            if Winner:
                genoms[index][1].fitness += 1000
                try:
                    print(Player.Iter, "На уровень")
                except:pass

                Learn = False
                break

        for _ in range(3):
            try:
                M = genoms[0][1].fitness
                MIndex = 0
                Ma = genoms[0][1].fitness
                for index in range(len(Players)):
                    Rate = genoms[index][1].fitness
                    if Rate < M: M = Rate; MIndex = index
                    Ma = Rate if Rate > Ma else Ma
                try:
                    print(M, Ma, round(200 / len(Players) / 2), "%")
                except:pass
                
                Loh = Players[MIndex]
                Loh.CanMove = False
                Loh.hideturtle()
                Players.pop(MIndex)
                nets.pop(MIndex)
                del Loh
            except IndexError:pass

    for Index, i in enumerate(Players):
        i.hideturtle()
        Players.pop(Index)
    
    gc.collect()

try:
    Last = ""
    for i in os.listdir(os.getcwd()):
        if "dataGen" in i:
            Last = i
    p = neat.Checkpointer.restore_checkpoint(Last)
except:
    p = neat.Population(config)
    print("SomeError")
p.add_reporter(neat.Checkpointer(1, 5, "dataGen"))
try:
    p.run(StartLearn)
except: print(traceback.format_exc())
exitonclick()