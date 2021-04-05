 # -*- coding: utf-8 -*-
 
""" Segmentar imágenes del microscopio a través del histograma"""

from skimage import io, img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage as nd

img = io.imread("./Imagenes/segmentacion/BSE_25sigma_noisy.jpg")

# Cambiar el tipo de dato uint8 a float
float_img = img_as_float(img)

# En vez de definir arbitrariamente el sigma, se puede determinar de esta manera
sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))

denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                               patch_size=5, patch_distance=3, multichannel=True)

# Nuevamente se realiza cambio de dato -> Me sirve simplemente para visualizar la imagen
denoise_img_as_8byte = img_as_ubyte(denoise_img)

# Arbitrariamente podemos definir segmentos arbitrarios
segm1 = (denoise_img_as_8byte <= 57)
segm2 = (denoise_img_as_8byte > 57) & (denoise_img_as_8byte <= 110)
segm3 = (denoise_img_as_8byte > 110) & (denoise_img_as_8byte <= 210)
segm4 = (denoise_img_as_8byte > 210)

# Para mostrar las imágenes -> Elimina el ruido del tamaño de la imagen, pero en blanco
all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3))

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)
all_segments[segm3] = (0,0,1)
all_segments[segm4] = (1,1,0)
plt.imshow(all_segments)
plt.show()

""" 
Muchos puntos amarillos, puntos rojos y puntos perdidos. ¿cómo limpiar la imagen? 
Podemos utilizar operaciones binarias de opening y closing: 
El Opening se encarga de los píxeles aislados dentro de la ventana
El Closing se encarga de los agujeros aislados dentro de la ventana definida
"""

segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))

all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but 714, 901, 3

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)

plt.imshow(all_segments_cleaned)  # Todo el ruido debería limpiarse ahora
plt.show()
