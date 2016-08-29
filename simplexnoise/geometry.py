"""
    Meant to help with ease of calculation with a coordinate system.
"""


class Point(object):
    """ Represents a point in a 3D dimensional system """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def as_list(self):
        return [self.x, self.y, self.z]

    def as_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def dot(self, x, y, z):
        return self.x * x + self.y * y + self.z * z

    def dot_point(self, x, y, z):
        return self.dot(point.x, point.y, point.z)

    def add(self, x, y, z):
        return Point(self.x + x, self.y + y, self.z + z)

    def add_point(self, point):
        return self.add(point.x, point.y, point.z)

    def subtract(self, x, y, z):
        return Point(self.x - x, self.y - y, self.z - z)

    def subtract_point(self, point):
        return self.subtract(point.x, point.y, point.z)
