import math
import csv
import sys
import random
import numpy.random as rand
import numpy as np
from matplotlib import pyplot as plt
from time import time


class SimAnneal:
    def __init__(self, f, q, x, startTemp, endTemp, coolFactor):
        self.f = f  # The function to be maximized
        self.q = q  # The function for chooising new configurations
        self.startX = x  # The starting configuration
        self.startTemp = startTemp
        self.endTemp = endTemp
        self.coolFactor = coolFactor

    def optimize(self):
        temp = self.startTemp
        curX = self.startX
        curCost = self.f(curX)
        optimalX = self.startX
        optimalCost = 0
        nSteps = 0

        self.xVals = []
        self.minimums = []
        while temp > self.endTemp:
            newX = self.q(curX.copy(), temp)
            newCost = self.f(newX)
            diff = newCost - curCost

            print(newCost, "NEW Cost")
            # This is because we are just looking for the first rate that beats 0.
            if newCost <= 0:
                break

            if diff > 0 or math.exp(diff / temp) > rand.uniform(0, 1):
                curX = newX  # Shallow copy is fine here
                curCost = 0

                # Now, if the new cost is lower than the best we found, record it
                # NOTE THE ONE BIG CHANGE HERE ... we need to copy the list for
                # recording the optimalX otherwise it continues to change as we
                # permute because it would make a copy of the REFERENCE to the list
                if newCost > optimalCost:
                    optimalX = curX.copy()
                    # optimalCost = curCost

            self.xVals.append(curX.copy())
            self.minimums.append(optimalX)
            temp = temp - self.coolFactor

            nSteps += 1
            if nSteps % 5000 == 0:
                curX = optimalX.copy()
                curCost = optimalCost

        return optimalX


# Goal is to minimize the savings rate while still getting the portfolio to last
# 60 years with all the same settings.
