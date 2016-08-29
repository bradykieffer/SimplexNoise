import numpy as np
from PIL import Image
from simplexnoise.noise import SimplexNoise, normalize

size = 250
sn = SimplexNoise(num_octaves=7, persistence=0.1, dimensions=2)
data = []

for i in xrange(size):
    data.append([])
    for j in xrange(size):
        noise = normalize(sn.fractal(i, j, hgrid=size))
        data[i].append(noise * 255.0)

# Cast to numpy array so we can save 
data = np.array(data).astype(np.uint8)
img = Image.fromarray(data, mode='L')
img.save('./fbm_example.png')