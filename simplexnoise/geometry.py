"""
    Meant to help with ease of calculation with a coordinate system.
"""

class Point2D(object):
    """ Represents a point on a 2D grid """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_tuple(self):
        return (self.x, self.y)

    def as_list(self):
        return [self.x, self.y]

    def as_dict(self):
        return {'x': self.x, 'y': self.y}

    def dot(self, x, y):
        return self.x * x + self.y * y

    def dot_point(self, x, y):
        return self.dot(point.x, point.y)

    def add(self, x, y):
        return Point2D(self.x + x, self.y + y)

    def add_point(self, point):
        return self.add(point.x, point.y)

    def subtract(self, x, y):
        return Point2D(self.x - x, self.y - y)

    def subtract_point(self, point):
        return self.subtract(point.x, point.y)