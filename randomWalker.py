 # -*- coding: utf-8 -*-
 
"""Para cuando hay 2 regiones con poco contraste entre sí y tengo ruido (salt & pepper)"""
 
import matplotlib.pyplot as plt
from skimage import io, img_as_float, exposure
from skimage.restoration import denoise_nl_means, estimate_sigma 
from skimage.segmentation import random_walker
from scipy import ndimage as nd
import numpy as np

img = img_as_float(io.imread("./Imagenes/ruido/Alloy_noisy.jpg"))
#plt.hist(img.flat, bins=100, range=(0, 1))
#plt.show()

# Eliminar el ruido
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3, multichannel=True)
#plt.hist(denoise_img.flat, bins=100, range=(0, 1)) 
#plt.show()

"""
Muchas veces no se pueden segmentar las regiones tan fácilmente porque son muy similares.
Para resolver eso se utiliza histograma de equalización.
Es más agresivo equalize_hist que equalize_adapthist 
"""

#eq_img = exposure.equalize_hist(denoise_img)
eq_img = exposure.equalize_adapthist(denoise_img)
plt.imshow(eq_img, cmap='gray')
#plt.hist(eq_img.flat, bins=100, range=(0., 1))
plt.show()

# Para definir esos valores (0.8, 1.7, 0.85, etc) me fijo donde están los picos del histograma
# Esos valores están entre 0 y 1
markers = np.zeros(img.shape, dtype=np.uint)
markers[(eq_img < 0.8) & (eq_img > 0.7)] = 1
markers[(eq_img > 0.85) & (eq_img < 0.99)] = 2

# Segmentación con el algoritmo Random Walker
labels = random_walker(eq_img, markers, beta=10, mode='bf')
segm1 = (labels == 1)
segm2 = (labels == 2)
all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)
plt.imshow(all_segments)
plt.show()

# Para obtener una imagen sin ruido, proceder a hacer lo siguiente
segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))

all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 
all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)

plt.imshow(all_segments_cleaned) 
plt.show()
