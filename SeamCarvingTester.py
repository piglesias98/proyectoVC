# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""
import cv2
import sys

import numpy as np

from matplotlib import pyplot as plt


import SeamCarving




def representar_imagenes(lista_imagen_leida, lista_titulos, n_col=2, tam=15):

    # Comprobamos que el numero de imágenes corresponde con el número de títulos pasados
	if len(lista_imagen_leida) != len(lista_titulos):
		print("No hay el mismo número de imágenes que de títulos.")
		return -1 # No hay el mismo numero de imágenes que de títulos

	# Calculamos el numero de imagenes
	n_imagenes = len(lista_imagen_leida)

	# Calculamos el número de filas
	n_filas = (n_imagenes // n_col) + (n_imagenes % n_col)

	# Establecemos por defecto un tamaño a las imágenes
	plt.figure(figsize=(tam,tam))

	# Recorremos la lista de imágenes
	for i in range(0, n_imagenes):

		plt.subplot(n_filas, n_col, i+1) # plt.subplot empieza en 1

		if (len(np.shape(lista_imagen_leida[i]))) == 2: # Si la imagen es en gris
			plt.imshow(lista_imagen_leida[i], cmap = 'gray')
		else: # Si la imagen es en color
			plt.imshow(cv2.cvtColor(lista_imagen_leida[i], cv2.COLOR_BGR2RGB))

		plt.title(lista_titulos[i]) # Añadimos el título a cada imagen

		plt.xticks([]), plt.yticks([]) # Para ocultar los valores de tick en los ejes X e Y

	plt.show()





# Prueba de funcionamiento
image = SeamCarving.readImage("surfista-min.jpeg", 1)

img = image.copy()

simple_energy = SeamCarving.simpleEnergy(img)
simple_energy2 = SeamCarving.simpleEnergyRGB(img)

#img2 = cv2.cvtColor(simple_energy, cv2.COLOR_BGR2RGB)
#simple_energy2 = cv2.cvtColor(simple_energy2, cv2.COLOR_BGR2RGB)

representar_imagenes([simple_energy, simple_energy2], ["energia1", "energia1"])


#cv2.imshow("original", image)
#cv2.imshow("energia 1", simple_energy)
#cv2.imshow("energia 2", simple_energy2)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#T, options = seamsOrder(image, image.shape[0], image.shape[1]-100)
#order = selectSeamsOrder (image, T, options)

#img1 = addOrderSeams (image, order)
#img2 = removeOrderSeams (image, order)