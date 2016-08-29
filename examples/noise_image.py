import numpy as np
from PIL import Image
from simplexnoise.noise import SimplexNoise, normalize

size = 250
noise_scale = 700.0 # Turns up the contrast 
sn = SimplexNoise(num_octaves=7, persistence=0.1, dimensions=2, noise_scale=noise_scale)
data = []

for i in xrange(size):
    data.append([])
    for j in xrange(size):
        noise = normalize(sn.noise(i, j))
        data[i].append(noise * 255.0)

# Cast to numpy array so we can save 
data = np.array(data).astype(np.uint8)
img = Image.fromarray(data, mode='L')
img.save('./noise_example.png')