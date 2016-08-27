import numpy as np
from PIL import Image
from simplexnoise.noise import SimplexNoise

size = 250
sn = SimplexNoise(num_octaves=7, persistence=0.1)
data = []

for i in xrange(size):
    data.append([])
    for j in xrange(size):
        noise = ((1 + sn.fractal(i, j, hgrid=size)) / 2.0)
        if noise > 1:
            noise = 1
        if noise < 0:
            noise = 0
        data[i].append(noise * 255.0)

# Cast to numpy array so we can save 
data = np.array(data).astype(np.uint8)
img = Image.fromarray(data, mode='L')
img.save('./fbm_example.png')