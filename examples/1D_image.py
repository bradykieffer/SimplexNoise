import matplotlib.pyplot as plt
from simplexnoise.noise import PerlinNoise, normalize

length = 10000
pn = PerlinNoise(num_octaves=7, persistence=0.1)
data = []

t = [i for i in xrange(length)]
for i in xrange(length):
    data.append(normalize(pn.fractal(x=i, hgrid=length)))

fig = plt.figure()
plt.plot(t, data)
fig.savefig('1D_example.png')
