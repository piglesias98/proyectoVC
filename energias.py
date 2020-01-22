# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import hog

# Referencias:
# -> http://pages.cs.wisc.edu/~moayad/cs766/index.html
# -> https://medium.com/swlh/real-world-dynamic-programming-seam-carving-9d11c5b0bfca
# -> https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
'''
Energía simple

Derivada en x e y de la imagen

Entrada:
    -> image: imagen

Salida:
    -> Matriz de energía
'''
def simpleEnergy (image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float)

    x = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
    y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3))

    return x + y

'''
Energía simple

Calcula la energía para cada canal r g b

Entrada:
    -> image: imagen

Salida:
    -> Matriz de energía
'''
def simpleEnergyRGB(image):
    b, g, r = cv2.split(image)
    b_energy = np.absolute(cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3))
    g_energy = np.absolute(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)) + np.absolute(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
    r_energy = np.absolute(cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)) + np.absolute(cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3))

    return b_energy + g_energy + r_energy

'''
Energía eHOG

Entrada:
    -> image: imagen

Salida:
    -> Matriz de energía
'''
def eHOG(image):

    simple_energy = simpleEnergyRGB(image)

    hogg = hog(image, orientations = 8, pixels_per_cell=(11,11), cells_per_block=(1,1), block_norm='L1',feature_vector=False, multichannel=True)

    energy = np.zeros((image.shape[0],image.shape[1]))
    maxHOG = np.zeros((hogg.shape[0], hogg.shape[1]))

    for i in range(hogg.shape[0]):
        for j in range(hogg.shape[1]):

            maxHOG[i,j] = max(hogg[i,j,0,0]) * 5000


    for i in range (maxHOG.shape[0]):
        for j in range (maxHOG.shape[1]):

            for k in range (11):
                for l in range (11):
                    indx = 11*i + k
                    indy = 11*j + l

                    energy[indx, indy] = simple_energy[indx, indy] / maxHOG[i,j]

    return energy

'''
Energía "forward"

Entrada:
    -> image: imagen

Salida:
    -> Matriz de energía 
'''
def forwardEnergy(image):

    n, m = image.shape[:2]

    energy = np.zeros((n,m))
    M = np.zeros((n,m))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    img = cv2.copyMakeBorder(img, top=1, bottom=0, left=1, right=1, borderType=cv2.BORDER_REPLICATE)

    U = np.roll(img, 1, axis=0)
    L = np.roll(img, 1, axis=1)
    R = np.roll(img, -1, axis=1 )

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    cU = cU[1:, 1:-1]
    cL = cL[1:, 1:-1]
    cR = cR[1:, 1:-1]

    M[0] = cU[0]

    for i in range(1, n):

        mU = M[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        # M(x,y) = min {M(x-1,y-1) + CL(x,y)
        #               M(x,y-1) + CU(x,y)
        #               M(x+1,y+1) + CR(x,y)}

        mLUR = np.array([mL, mU, mR])
        cLUR = np.array([cL[i], cU[i], cR[i]])
        mLUR += cLUR

        argmins = np.argmin(mLUR, axis=0)
        M[i] = np.choose(argmins, mLUR)
        energy[i] = np.choose(argmins, cLUR)

    # Devolvemos energía para crear el camino que implique una menor energía final (después de eliminar o añadir)
    return (energy)
