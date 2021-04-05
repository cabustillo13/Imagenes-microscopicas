# -*- coding: utf-8 -*-

"""Eliminando ruido de una imagen mediante filtros"""

from skimage import io, img_as_float
from scipy import ndimage as nd
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
import numpy as np

""" IMPORTANTE: Como realizaremos operaciones matemáticas con las imágenes
Es una buena práctica definir las imagenes como float"""
img = img_as_float(io.imread("./Imagenes/ruido/noisy_img.jpg"))

# Filtro Gaussiano
gaussian_img = nd.gaussian_filter(img, sigma=3)
plt.imshow(gaussian_img, cmap="gray")
plt.show()

# Filtro Mediana
median_img = nd.median_filter(img, size=3)
plt.imshow(median_img, cmap="gray")
plt.show()

# Filtro Denoising
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                               patch_size=5, patch_distance=3, multichannel=True)
plt.imshow(denoise_img, cmap="gray")
plt.show()
