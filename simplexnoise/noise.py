from __future__ import division
import math
import random
from geometry import Point2D


class SimplexNoiseOctave(object):

    # These allow us to skew (x,y) space and determine which simplex we are in
    # and then return to (x,y) space
    skew_factor = 0.5 * (math.sqrt(3.0) - 1.0)
    unskew_factor = (3.0 - math.sqrt(3.0)) / 6.0

    def __init__(self, num_shuffles=100):
        self.p_supply = [i for i in xrange(0, 256)]

        self.grads = [
            Point2D(1, 1),
            Point2D(-1, 1),
            Point2D(1, -1),
            Point2D(1, -1)
        ]

        for i in xrange(num_shuffles):
            random.shuffle(self.p_supply)

        self.perm = self.p_supply * 2
        self.perm_mod_4 = [i % 4 for i in self.perm]

    def noise(self, xin, yin, noise_scale):
        # Point lists
        points_xy = []
        points_ij = []

        # Get the skewed coordinates
        points_ij.append(self.skew_cell(xin, yin))
        t = (points_ij[0].x + points_ij[0].y) * self.unskew_factor

        # Unskew the cell back to (x, y) space
        pt_X0_Y0 = self.unskew_cell(points_ij[0], t)
        points_xy.append(Point2D(xin - pt_X0_Y0.x, yin - pt_X0_Y0.y))

        # In 2D the simplex is an equi. triangle. Determine which simplex we are in
        # The coords are either (0, 0), (1, 0), (1, 1)
        # or they are:          (0, 0), (0, 1), (1, 1)
        points_ij.append(self.determine_simplex(points_xy[0]))

        for pt in self.get_simplex_coords(points_xy[0], points_ij[1]):
            points_xy.append(pt)

        # Hashed gradient indices of the three simplex corners
        grad_index_hash = self.hashed_gradient_indices(points_ij)

        # Calculate the contributions from the three corners
        noise_contribs = self.calc_noise_contributions(
            grad_index_hash, points_xy)

        # Move our range to [-1, 1]
        return noise_scale * sum(noise_contribs)

    def skew_cell(self, xin, yin):
        """ Skew the input space and determine which simplex we are in """
        skew = (xin + yin) * self.skew_factor
        i = self._fast_floor(xin + skew)  # int(math.floor(xin + skew))
        j = self._fast_floor(yin + skew)  # int(math.floor(yin + skew))

        return Point2D(i, j)

    def unskew_cell(self, pt, t):
        """ Return to (x,y) space """
        return Point2D(pt.x - t, pt.y - t)

    def determine_simplex(self, pt):
        if pt.x > pt.y:
            # Lower triangle -> (0, 0), (1, 0), (1, 1)
            return Point2D(1, 0)

        # Upper triangle -> (0, 0), (0, 1), (1, 1)
        return Point2D(0, 1)

    def get_simplex_coords(self, pt, pt_ij):
        """
            A step of (1,0) in (i,j) means a step of (1-unskew_factor,-unskew_factor) in (x,y), and
            a step of (0,1) in (i,j) means a step of (-unskew_factor,1-unskew_factor) in (x,y)
            Now get the other (x, y) coordinates following this logic
        """

        x1 = pt.x - pt_ij.x + self.unskew_factor
        y1 = pt.y - pt_ij.y + self.unskew_factor

        # Last corners in skewed coords
        x2 = pt.x - 1.0 + 2.0 * self.unskew_factor
        y2 = pt.y - 1.0 + 2.0 * self.unskew_factor
        return [Point2D(x1, y1), Point2D(x2, y2)]

    def hashed_gradient_indices(self, points_ij):
        ii = points_ij[0].x & 255
        jj = points_ij[0].y & 255

        return (
            self.perm_mod_4[ii + self.perm[jj]],
            self.perm_mod_4[ii + points_ij[1].x +
                            self.perm[jj + points_ij[1].y]],
            self.perm_mod_4[ii + 1 + self.perm[jj + 1]]
        )

    def calc_noise_contributions(self, grad_index_hash, points_xy):
        """ Calculates the contribution from each corner (in 2D there are three!) """
        contribs = []
        for i in xrange(len(grad_index_hash)):
            x = points_xy[i].x
            y = points_xy[i].y
            grad = self.grads[grad_index_hash[i]]
            t = 0.5 - x * x - y * y

            if t < 0:
                contribs.append(0)
            else:
                t *= t
                contribs.append(t * t * grad.dot(x, y))

        return contribs

    def _fast_floor(self, x):
        xi = int(x)
        if x < xi:
            return xi - 1
        return xi


class SimplexNoise(object):

    def __init__(self, num_octaves, persistence):
        self.num_octaves = num_octaves
        self.octaves = [SimplexNoiseOctave() for i in xrange(self.num_octaves)]
        self.frequencies = [pow(2, i) for i in xrange(self.num_octaves)]
        self.amplitudes = [pow(persistence, len(self.octaves) - i)
                           for i in xrange(self.num_octaves)]

    def noise(self, x, y, noise_scale=70.0):
        noise = [self.octaves[i].noise(x / self.frequencies[i], y / self.frequencies[
                                       i], noise_scale) * self.amplitudes[i] for i in xrange(self.num_octaves)]
        return sum(noise)

    def fractal(self, x, y, hgrid, lacunarity=2.0, gain=0.65, noise_scale=70.0):
        """ A more refined approach but has a much slower run time """
        noise = []
        frequency = 1.0 / hgrid
        amplitude = gain
        for i in xrange(self.num_octaves):
            noise.append(self.octaves[i].noise(
                x * frequency, y * frequency, noise_scale) * amplitude)
            frequency *= lacunarity
            amplitude *= gain

        return sum(noise)
