""" Unit tests for the geometry package """
import nose
from simplexnoise.geometry import Point
from nose.tools import assert_equal


class TestPoint(object):

    def setUp(self):
        self.point = Point(1, 1, 1)

    def test_as_tuple(self):
        tup = (1, 1, 1)
        assert_equal(self.point.as_tuple(), tup)

    def test_as_list(self):
        ls = [1, 1, 1]
        assert_equal(self.point.as_list(), ls)

    def test_as_dict(self):
        d = {'x': 1, 'y': 1, 'z': 1}
        assert_equal(self.point.as_dict(), d)

    def test_dot(self):
        assert_equal(self.point.dot(1, -1, 0), 0)

    def test_add(self):
        pt = self.point.add(1, -1, 2)
        assert_equal(pt.x, 2)
        assert_equal(pt.y, 0)
        assert_equal(pt.z, 3)

    def test_subtract(self):
        pt = self.point.subtract(-1, 1, -2)
        assert_equal(pt.x, 2)
        assert_equal(pt.y, 0)
        assert_equal(pt.z, 3)
