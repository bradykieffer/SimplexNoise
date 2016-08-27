import math
import random
from geometry import Point2D

class SimplexNoiseOctave(object):

    def __init__(self, num_shuffles=100):
        self.p_supply = [i for i in xrange(0, 256)]
        
        # Drawn like this
        #  2--1
        #  |  |
        #  3--4
        self.points = [ Point2D(1, 1), Point2D(-1, 1), Point2D(-1, -1), Point2D(1, -1) ]

        for i in xrange(num_shuffles):
            random.shuffle(self.p_supply)

        self.perm = self.p_supply * 2
        self.perm_mod_12 = [i % 12 for i in self.perm]

    def _fast_floor(self, x):
        xi = int(x)
        if x < xi:
            return xi - 1
        return xi

    def noise(self, xin, yin):
        pass


class SimplexNoise(object):
    def __init__(self, num_octaves, precision):
        self.octaves = [0 for i in xrange(num_octaves)]
        self.frequencies = [pow(2, i) for i in xrange(num_octaves)]
        self.amplitudes = [pow(precision, i) for i in xrange(num_octaves)]
