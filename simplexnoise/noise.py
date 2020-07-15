from __future__ import division
import math
import random
from .geometry import Point

# Constants to avoid magic numbers
DEFAULT_NOISE_SCALE = -1  # Check noise_scale against this
DEFAULT_1D_NOISE_SCALE = 0.188
DEFAULT_2D_NOISE_SCALE = 70.0
DEFAULT_3D_NOISE_SCALE = 32.0
DEFAULT_LACUNARITY = 2.0
DEFAULT_GAIN = 0.65
DIMENSIONS_2D = 2
DIMENSIONS_3D = 3
DEFAULT_SHUFFLES = 100


def normalize(x):
    res = (1.0 + x) / 2.0

    # Clamp the result, this is not ideal
    if res > 1:
        res = 1
    if res < 0:
        res = 0

    return res


class PerlinNoise(object):
    """ 
        Implementation of 1D Perlin Noise ported from C code: 
        https://github.com/stegu/perlin-noise/blob/master/src/noise1234.c
    """

    def __init__(self, num_octaves, persistence, noise_scale=DEFAULT_NOISE_SCALE):
        self.num_octaves = num_octaves

        if DEFAULT_NOISE_SCALE == noise_scale:
            self.noise_scale = DEFAULT_1D_NOISE_SCALE
        else:
            self.noise_scale = noise_scale

        self.octaves = [PerlinNoiseOctave() for i in range(self.num_octaves)]
        self.frequencies = [1.0 / pow(2, i) for i in range(self.num_octaves)]
        self.amplitudes = [pow(persistence, len(self.octaves) - i)
                           for i in range(self.num_octaves)]

    def noise(self, x):
        noise = [
            self.octaves[i].noise(
                xin=x * self.frequencies[i],
                noise_scale=self.noise_scale
            ) * self.amplitudes[i] for i in range(self.num_octaves)]

        return sum(noise)

    def fractal(self, x, hgrid, lacunarity=DEFAULT_LACUNARITY, gain=DEFAULT_GAIN):
        """ A more refined approach but has a much slower run time """
        noise = []
        frequency = 1.0 / hgrid
        amplitude = gain

        for i in range(self.num_octaves):
            noise.append(
                self.octaves[i].noise(
                    xin=x * frequency,
                    noise_scale=self.noise_scale
                ) * amplitude
            )

            frequency *= lacunarity
            amplitude *= gain

        return sum(noise)


class PerlinNoiseOctave(object):

    def __init__(self, num_shuffles=DEFAULT_SHUFFLES):
        self.p_supply = [i for i in range(0, 256)]

        for i in range(num_shuffles):
            random.shuffle(self.p_supply)

        self.perm = self.p_supply * 2

    def noise(self, xin, noise_scale):
        ix0 = int(math.floor(xin))
        fx0 = xin - ix0
        fx1 = fx0 - 1.0
        ix1 = (ix0 + 1) & 255
        ix0 = ix0 & 255

        s = self.fade(fx0)

        n0 = self.grad(self.perm[ix0], fx0)
        n1 = self.grad(self.perm[ix1], fx1)

        return noise_scale * self.lerp(s, n0, n1)

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def fade(self, t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def grad(self, hash, x):
        h = hash & 15
        grad = 1.0 + (h & 7)  # Gradient value from 1.0 - 8.0
        if h & 8:
            grad = -grad  # Add a random sign
        return grad * x


class SimplexNoise(object):

    def __init__(self, num_octaves, persistence, dimensions, noise_scale=DEFAULT_NOISE_SCALE):
        self.num_octaves = num_octaves
        dimensions = int(dimensions)

        if DIMENSIONS_2D == dimensions:
            self.octaves = [SimplexNoiseOctave2D()
                            for i in range(self.num_octaves)]
            self.noise_scale = DEFAULT_2D_NOISE_SCALE

        elif DIMENSIONS_3D == dimensions:
            self.octaves = [SimplexNoiseOctave3D()
                            for i in range(self.num_octaves)]
            self.noise_scale = DEFAULT_2D_NOISE_SCALE

        else:
            raise ArgumentError(
                'Please supply the dimensions of noise generation (2 or 3)'
            )

        if DEFAULT_NOISE_SCALE != noise_scale:
            self.noise_scale = noise_scale

        self.frequencies = [pow(2, i) for i in range(self.num_octaves)]
        self.amplitudes = [pow(persistence, len(self.octaves) - i)
                           for i in range(self.num_octaves)]

    def noise(self, x=0, y=0, z=0):
        noise = [
            self.octaves[i].noise(
                xin=x / self.frequencies[i],
                yin=y / self.frequencies[i],
                zin=z / self.frequencies[i],
                noise_scale=self.noise_scale
            ) * self.amplitudes[i] for i in range(self.num_octaves)]

        return sum(noise)

    def fractal(self, x=0, y=0, z=0, hgrid=0, lacunarity=DEFAULT_LACUNARITY, gain=DEFAULT_GAIN):
        """ A more refined approach but has a much slower run time """
        noise = []
        frequency = 1.0 / hgrid
        amplitude = gain

        for i in range(self.num_octaves):
            noise.append(
                self.octaves[i].noise(
                    xin=x * frequency,
                    yin=y * frequency,
                    zin=z * frequency,
                    noise_scale=self.noise_scale
                ) * amplitude
            )

            frequency *= lacunarity
            amplitude *= gain

        return sum(noise)


class SimplexNoiseOctave2D(object):

    # These allow us to skew (x,y) space and determine which simplex we are in
    # and then return to (x,y) space
    skew_factor = 0.5 * (math.sqrt(3.0) - 1.0)
    unskew_factor = (3.0 - math.sqrt(3.0)) / 6.0

    def __init__(self, num_shuffles=DEFAULT_SHUFFLES):
        self.p_supply = [i for i in range(0, 256)]

        self.grads = [
            Point(1, 1, 0),
            Point(-1, 1, 0),
            Point(1, -1, 0),
            Point(-1, -1, 0)
        ]

        for i in range(num_shuffles):
            random.shuffle(self.p_supply)

        self.perm = self.p_supply * 2
        self.perm_mod_4 = [i % 4 for i in self.perm]

    def noise(self, xin, yin, zin, noise_scale):
        # Point lists
        points_xy = []
        points_ij = []

        # Get the skewed coordinates
        points_ij.append(self.skew_cell(xin, yin))
        t = (points_ij[0].x + points_ij[0].y) * self.unskew_factor

        # Unskew the cell back to (x, y) space
        pt_X0_Y0 = self.unskew_cell(points_ij[0], t)
        points_xy.append(Point(xin - pt_X0_Y0.x, yin - pt_X0_Y0.y, 0))

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
        i = int(math.floor(xin + skew))
        j = int(math.floor(yin + skew))

        return Point(i, j, 0)

    def unskew_cell(self, pt, t):
        """ Return to (x,y) space """
        return Point(pt.x - t, pt.y - t, 0)

    def determine_simplex(self, pt):
        if pt.x > pt.y:
            # Lower triangle -> (0, 0), (1, 0), (1, 1)
            return Point(1, 0, 0)

        # Upper triangle -> (0, 0), (0, 1), (1, 1)
        return Point(0, 1, 0)

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
        return (Point(x1, y1, 0), Point(x2, y2, 0))

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
        for i in range(len(grad_index_hash)):
            x = points_xy[i].x
            y = points_xy[i].y
            grad = self.grads[grad_index_hash[i]]
            t = 0.5 - x * x - y * y

            if t < 0:
                contribs.append(0)
            else:
                t *= t
                contribs.append(t * t * grad.dot(x, y, 0))

        return contribs


class SimplexNoiseOctave3D(object):
    # These allow us to skew (x,y) space and determine which simplex we are in
    # and then return to (x,y) space
    skew_factor = 1.0 / 3.0
    unskew_factor = 1.0 / 6.0

    def __init__(self, num_shuffles=DEFAULT_SHUFFLES):
        self.p_supply = [i for i in range(0, 256)]

        self.grads = [
            Point(1, 1, 0), Point(-1, 1, 0), Point(1, -1, 0), Point(-1, -1, 0),
            Point(1, 0, 1), Point(-1, 0, 1), Point(1, 0, -1), Point(-1, 0, -1),
            Point(0, 1, 1), Point(0, -1, 1), Point(0, 1, -1), Point(0, -1, -1),
        ]

        for i in range(num_shuffles):
            random.shuffle(self.p_supply)

        self.perm = self.p_supply * 2
        self.perm_mod_12 = [i % 12 for i in self.perm]

    def noise(self, xin, yin, zin, noise_scale):
        # Point lists
        points_xyz = []
        points_ijk = []

        # Get the skewed coordinates
        points_ijk.append(self.skew_cell(xin, yin, zin))
        t = (points_ijk[0].x + points_ijk[0].y +
             points_ijk[0].z) * self.unskew_factor

        # Unskew the cell back to (x, y) space
        pt_X0_Y0_Z0 = self.unskew_cell(points_ijk[0], t)
        points_xyz.append(
            Point(
                xin - pt_X0_Y0_Z0.x,
                yin - pt_X0_Y0_Z0.y,
                zin - pt_X0_Y0_Z0.z
            )
        )

        # In 3D the simplex is a slightly irregular tetrahedron
        for pt in self.determine_simplex(points_xyz[0]):
            points_ijk.append(pt)

        for pt in self.get_simplex_coords(points_xyz[0], points_ijk[1], points_ijk[2]):
            points_xyz.append(pt)

        # Hashed gradient indices of the three simplex corners
        grad_index_hash = self.hashed_gradient_indices(points_ijk)

        # Calculate the contributions from the three corners
        noise_contribs = self.calc_noise_contributions(
            grad_index_hash, points_xyz)

        # Move our range to [-1, 1]
        return noise_scale * sum(noise_contribs)

    def skew_cell(self, xin, yin, zin):
        """ Skew the input space and determine which simplex we are in """
        skew = (xin + yin + zin) * self.skew_factor
        i = int(math.floor(xin + skew))
        j = int(math.floor(yin + skew))
        k = int(math.floor(zin + skew))

        return Point(i, j, k)

    def unskew_cell(self, pt, t):
        """ Return to (x,y) space """
        return Point(pt.x - t, pt.y - t, pt.z - t)

    def determine_simplex(self, pt):
        if pt.x >= pt.y:
            if pt.y >= pt.z:
                return (Point(1, 0, 0), Point(1, 1, 0))
            elif pt.x >= pt.z:
                return (Point(1, 0, 0), Point(1, 0, 1))
            else:
                return (Point(0, 0, 1), Point(1, 0, 1))
        else:
            if pt.y < pt.z:
                return (Point(0, 0, 1), Point(0, 1, 1))
            elif pt.x < pt.z:
                return (Point(0, 1, 0), Point(0, 1, 1))
            else:
                return (Point(0, 1, 0), Point(1, 1, 0))

    def get_simplex_coords(self, pt, pt_ijk_1, pt_ijk_2):
        x1 = pt.x - pt_ijk_1.x + self.unskew_factor
        y1 = pt.y - pt_ijk_1.y + self.unskew_factor
        z1 = pt.z - pt_ijk_1.z + self.unskew_factor

        x2 = pt.x - pt_ijk_2.x + 2.0 * self.unskew_factor
        y2 = pt.y - pt_ijk_2.y + 2.0 * self.unskew_factor
        z2 = pt.z - pt_ijk_2.z + 2.0 * self.unskew_factor

        # Last corners in skewed coords
        x3 = pt.x - 1.0 + 3.0 * self.unskew_factor
        y3 = pt.y - 1.0 + 3.0 * self.unskew_factor
        z3 = pt.z - 1.0 + 3.0 * self.unskew_factor

        return (Point(x1, y1, z1), Point(x2, y2, z2), Point(x3, y3, z3))

    def hashed_gradient_indices(self, points_ijk):
        ii = points_ijk[0].x & 255
        jj = points_ijk[0].y & 255
        kk = points_ijk[0].z & 255

        return (
            self.perm_mod_12[ii + self.perm[jj + self.perm[kk]]],
            self.perm_mod_12[ii + points_ijk[1].x + self.perm[jj +
                                                              points_ijk[1].y + self.perm[kk + points_ijk[1].z]]],
            self.perm_mod_12[ii + points_ijk[2].x + self.perm[jj +
                                                              points_ijk[2].y + self.perm[kk + points_ijk[2].z]]],
            self.perm_mod_12[ii + 1 + self.perm[jj + 1 + self.perm[kk + 1]]]
        )

    def calc_noise_contributions(self, grad_index_hash, points_xyz):
        """ Calculates the contribution from each corner (in 2D there are three!) """
        contribs = []
        for i in range(len(grad_index_hash)):
            x = points_xyz[i].x
            y = points_xyz[i].y
            z = points_xyz[i].z
            grad = self.grads[grad_index_hash[i]]
            t = 0.6 - x * x - y * y - z * z

            if t < 0:
                contribs.append(0)
            else:
                t *= t
                contribs.append(t * t * grad.dot(x, y, z))

        return contribs
