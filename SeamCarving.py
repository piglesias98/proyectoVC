# -*- coding: utf-8 -*-

import cv2
import sys

import numpy as np

from matplotlib import pyplot as plt

# Leer la imagen de entrada
# Por defecto, las imagenes se leen a color
def readImage (filename, flagColor = 1):

    image = cv2.imread(filename, flagColor)

    if (image.size == 0):
        print('Error al leer la imagen')
        sys.exit(-1)

    return image

# Duda en la energía, creo que la energía simple es así (fórmula 1 - página 3
# del paper). No estoy segura de los parámetros (tamaño del kernel)
# Referenias:   
# -> http://pages.cs.wisc.edu/~moayad/cs766/index.html
# -> https://medium.com/swlh/real-world-dynamic-programming-seam-carving-9d11c5b0bfca
# -> https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
def simpleEnergy (image):


    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float)

    x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    gradient = np.abs(x) + np.abs(y)
    
    return gradient


# Costura óptima vertical
def verticalSeam (image):

    n = image.shape[0]
    m = image.shape[1]

    energy = simpleEnergy(image)

    M = energy.copy()  # Matriz para la energía acumulativa mínima

    # Recorremos la imagen desde la segunda fila hasta la última
    for i in range (1, n):

        M[i,0] = energy[i,0] + min(M[i-1, 0], M[i-1,1])

        for j in range (1, m-1):

            M[i,j] = energy[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])

        M[i,m-1] = energy[i,m-1] + min(M[i-1, m-2], M[i-1,m-1])
    
    min_ind = np.argmin(M[n-1])

    camino = [np.argmin(M[n-1])]
    
    # Camino de la costura, buscando la menor energía posible
    for i in range (1, n):

        indy = camino[-1]

        min_ind = indy
        min_value = M[n - i - 1, indy]

        if (indy - 1) > -1 and min_value > M[n - i - 1, indy - 1]:

            min_ind = indy - 1
            min_value = M[n - i - 1, indy - 1]

        if (indy + 1) < m and min_value > M[n - i - 1, indy + 1]:

            min_ind = indy + 1
            min_value = M[n - i - 1, indy + 1]

        camino.append(min_ind)

    return M[n-1, min_ind], min_ind, camino

# Costura óptima vertical
def horizontalSeam (image):

    n = image.shape[0]
    m = image.shape[1]

    energy = simpleEnergy(image)

    M = energy.copy()  # Matriz para la energía acumulativa mínima

    # Recorremos la imagen desde la segunda columna hasta la última
    for j in range (1, m):

        M[0,j] = energy[0,j] + min(M[0, j-1], M[1,j-1])

        for i in range (1, n-1):

            M[i,j] = energy[i,j] + min(M[i-1,j-1], M[i,j-1], M[i+1,j-1])

        M[n-1,j] = energy[n-1,j] + min(M[n-2, j-1], M[n-1,j-1])

    min_ind = np.argmin(M[:,m-1])
    
    camino = [np.argmin(M[:,m-1])]

    # Camino de la costura, buscando la menor energía posible
    for j in range (1, m):

        indx = camino[-1]

        min_ind = indx
        min_value = M[indx, m - j - 1]

        if (indx - 1) > -1 and min_value > M[indx - 1, m - j - 1]:

            min_ind = indx - 1
            min_value = M[indx - 1, m - j - 1]

        if (indx + 1) < n and min_value > M[indx + 1, m - j - 1]:

            min_ind = indx + 1
            min_value = M[indx + 1, m - j - 1]

        camino.append(min_ind)
        
    return M[min_ind, m-1], min_ind, camino

# Para eliminar los píxeles seleccionados en el camino de la costura simplemente
# desplazo la fila hacia arriba o la columna a la izquierda y elimino la última 
# fila/columna
def removeSeam (image, camino, vertical):
    
    n = image.shape[0]
    m = image.shape[1]
    
    if vertical:
    
        for i in range (0, n):
            
            for j in range (camino[i], m - 1):
                image[n - i - 1, j] = image[n - i - 1, j + 1]
        
        return np.delete(image, -1, 1)
    
    for i in range (0, m):
            
        for j in range (camino[i], n - 1):
            image[j, m - i - 1] = image[j + 1, m - i - 1]
        
    return np.delete(image, -1, 0)
    
    
# Buscamos el orden en el que hay que aplicar las costuras para conseguir una
# imagen n x m -> n' x m' (fórmula 6 - página 5 del paper)
# Primero he supuesto que solo vamos a reducir imágenes, para ampliar, en vez 
# de eliminar habría que duplicar los píxeles promediando con los vecinos que
# no estén en el camino de la costura
def seamsOrder (img, nn, nm):
    
    image = img.copy()
    
    n = image.shape[0]
    m = image.shape[1]
    
    c = n - nn
    r = m - nm 
    
    ultimo = 0 # Indica cual es la ´última costura (0 -> horiz, 1 -> vertical)
    
    T = np.zeros((c,r))
    
    # ¿Se supone que el último píxel se hacen las dos costuras? Si no, no me salen 
    # los cálculos (si en T(r,c) solo se hace una, faltará una costura horizontal 
    # o vertical para conseguir el tamaño objetivo).

"""    

    Me lié con esta parte, los índices me perdieron 
    
    # Rellenamos la primera columna y fila de la tabla 
    horizontal = image.copy()
    
    for i in range (1, c):
        
        min_energy, indx, camino = horizontalSeam(horizontal)
        
        T[i,0] = T[i-1,0] + min_energy
        
        horizontal = removeSeam(horizontal, camino, 0)
    
    vertical = image.copy()
    
    for i in range (1, r):
        
        min_energy, indx, camino = verticalSeam(vertical)
        
        T[0,i] = T[0,i-1] + min_energy
        
        vertical = removeSeam(vertical, camino, 1)
    
    horizontal = image.copy()
    
    # No estoy segura si habría que ir modificando asi la imagen. Como se tiene 
    # que rellenar la tabla para cada posible tamaño que puede tomar la imagen, 
    # tendrá que, a la fuerza, eliminar c filas y r columnas, pero no se si tendría
    # que hacerse así
    for i in range (1, c):
        
        min_energy, indx, camino = horizontalSeam(horizontal)
        
        horizontal = removeSeam(horizontal, camino, 0)
        
        vertical = horizontal.copy()
        
        for j in range (1, r):
            
            min_horizontal, ind_horizontal, camino1 = horizontalSeam(vertical)
            min_vertical, ind_vertical, camino2 = verticalSeam(vertical)
            
            T[i,j] = min(T[i-1,j] + min_horizontal, T[i, j-1] + min_vertical)
            
            vertical = removeSeam(vertical, camino2, 1)
            
    if [c,r] == T[c, r-1] + min_vertical:
        ultimo = 1
"""    
    # Sacamos el orden de costuras. En el paper dice que se enpieza en T(r,c) hasta
    # T(0,0), pero supongo que se aplican en orden inverso (Desde T(0,0) hasta T(r,c))
                
        
def drawSeams(vertical_seams, horizontal_seams, image):
    
    n = image.shape[0]
    m = image.shape[1]
    
    for x in vertical_seams:
        
        for i in range (0, n):
            
            image[n - i - 1, x[i], 0] = 0
            image[n - i - 1, x[i], 1] = 0
            image[n - i - 1, x[i], 2] = 255
    
    for y in horizontal_seams:
        
        for j in range (0, m):
            
            image[y[j], m - j - 1, 0] = 0
            image[y[j], m - j - 1, 1] = 0
            image[y[j], m - j - 1, 2] = 255
            
    return image      
            
# Prueba de funcionamiento
image = readImage("surfista.jpeg", 1)

a,b,caminox = verticalSeam(image)
a,b,caminoy = horizontalSeam(image)

img2 = drawSeams([caminox], [caminoy], image.copy())

img3 = removeSeam (image, caminox, 1)

mini = a

for i in range (0, 50):
    a,b,caminox = verticalSeam(img3)
    
    img2 = drawSeams([caminox], [], img2)
        
    img3 = removeSeam (img3, caminox, 1)
    
# Tarda muchisimo en ejecutar con esta imagen porque es grande.
# El resultado no es el que tiene que ser, falta refinamiento (es solo para ver 
# que funciona el método)

#cv2.imshow("original", image)
#cv2.imshow("costuras", img2)
#cv2.imshow("costuras eliminadas", img3)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    
plt.title("original")
plt.imshow(img1)
plt.show()

plt.title("costuras elimindas")
plt.imshow(img3)
plt.show()

plt.title("costuras")
plt.imshow(img2)
plt.show()