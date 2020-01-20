# -*- coding: utf-8 -*-

import cv2
import sys

import numpy as np

from matplotlib import pyplot as plt

from skimage import exposure
from skimage import feature

from skimage.feature import hog

# Leer la imagen de entrada
# Por defecto, las imagenes se leen a color
def readImage (filename, flagColor = 1):

    image = cv2.imread(filename, flagColor)

    # Esto se comprueba de otra forma, tengo que buscarlo
    if (image.size == 0):
        print('Error al leer la imagen')
        sys.exit(-1)

    return image

def maskSize(mask):
    rows, cols = np.where(mask < 200)
    height = np.amax(rows) - np.amin(rows) + 1
    width = np.amax(cols) - np.amin(cols) + 1
    return height, width



#Prueba
# Duda en la energía, creo que la energía simple es así (fórmula 1 - página 3
# del paper). No estoy segura de los parámetros (tamaño del kernel)
# Referencias:
# -> http://pages.cs.wisc.edu/~moayad/cs766/index.html
# -> https://medium.com/swlh/real-world-dynamic-programming-seam-carving-9d11c5b0bfca
# -> https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
def simpleEnergy (image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float)

    x = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5))
    y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))

    return x + y

"""
    Calcula la energía para cada canal r g b
    No se aprecia diferencia en el resultado, no sé si es lo mismo

"""
def simpleEnergyRGB(image):
    b, g, r = cv2.split(image)
    b_energy = np.absolute(cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=5)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=5))
    g_energy = np.absolute(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)) + np.absolute(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5))
    r_energy = np.absolute(cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=5)) + np.absolute(cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=5))
    return b_energy + g_energy + r_energy


    
    
#def forwardEnergy(image):
#    n, m = image.shape[:2]
#    
#    energy = np.zeros((n,m))
#    M = np.zeros((n,m))
#    
#    U = np.roll(image, 1, axis=0)
#    L = np.roll(image, 1, axis=1)
#    r = np.roll(image, -1, axis=1 )
#    

def eHOG(image, funcion=0):

    if funcion:
        simple_energy = simpleEnergy(image)
    else:
        simple_energy = simpleEnergyRGB(image)

    hogg = hog(image, orientations = 9, pixels_per_cell=(11,11), cells_per_block=(1,1), feature_vector=True, multichannel=True)

    energy = np.zeros(image.shape[:2])
#    maxHOG = np.zeros((hogg.shape[0], hogg.shape[1]))
    bins = np.split (hogg, 9)

    bins = np.array(bins)

    bins = bins.max(axis=0)
    print(bins.shape)
#    for i in range(hogg.shape[0]):
#        for j in range(hogg.shape[1]):
#
#            # Multiplicamos por 255 para "desnormalización"
#            maxHOG[i,j] = max(hogg[i,j,0,0]) * 255



    # Calculamos eHOG
    for i in range (bins.shape[0]):

        for k in range (11):
            for l in range (11):
                indx = 11*i + k
                indy = 11*i + l

                energy[indx, indy] = simple_energy[indx, indy] / bins[i]

    return energy.astype(np.uint8)

def crearCamino (M):

    n, m = M.shape[:2]

    camino = [np.argmin(M[-1])] # Array de m elementos

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

    return camino

def forwardEnergy(image):

    n, m = image.shape[:2]

    energy = np.zeros((n,m))
    M = np.zeros((n,m))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

#    img = cv2.copyMakeBorder(img, top=1, bottom=0, left=1, right=1, borderType=cv2.BORDER_REPLICATE)

    U = np.roll(img, 1, axis=0)
    L = np.roll(img, 1, axis=1)
    R = np.roll(img, -1, axis=1 )

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

#    cU = cU[1:, 1:-1]
#    cL = cL[1:, 1:-1]
#    cR = cR[1:, 1:-1]

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

# Costura óptima vertical
def verticalSeam (image, funcion):

    n, m = image.shape[:2]

    energy = funcion(image)

    M = energy.copy()  # Matriz para la energía acumulativa mínima

    # Recorremos la imagen desde la segunda fila hasta la última
    for i in range (1, n):

        if m > 1:

            M[i,0] = energy[i,0] + min(M[i-1, 0], M[i-1,1])

        else:
            M[i,0] = energy[i,0] + M[i-1, 0]

        for j in range (1, m-1):

            M[i,j] = energy[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])

        M[i,m-1] = energy[i,m-1] + min(M[i-1, m-2], M[i-1,m-1])

    return crearCamino (M)

# Costura óptima vertical
def horizontalSeam (image):

    return verticalSeam(image)

#    n = image.shape[0]
#    m = image.shape[1]
#
#    energy = simpleEnergy(image)
#
#    M = energy.copy()  # Matriz para la energía acumulativa mínima
#
#    # Recorremos la imagen desde la segunda columna hasta la última
#    for j in range (1, m):
#
#        if n > 1:
#            M[0,j] = energy[0,j] + min(M[0, j-1], M[1,j-1])
#
#        else:
#            M[0,j] = energy[0,j] + M[0, j-1]
#
#        for i in range (1, n-1):
#
#            M[i,j] = energy[i,j] + min(M[i-1,j-1], M[i,j-1], M[i+1,j-1])
#
#        M[n-1,j] = energy[n-1,j] + min(M[n-2, j-1], M[n-1,j-1])
#
#    min_ind = np.argmin(M[:,m-1])
#
#    camino = [np.argmin(M[:,m-1])]
#
#    # Camino de la costura, buscando la menor energía posible
#    for j in range (1, m):
#
#        indx = camino[-1]
#
#        min_ind = indx
#        min_value = M[indx, m - j - 1]
#
#        if (indx - 1) > -1 and min_value > M[indx - 1, m - j - 1]:
#
#            min_ind = indx - 1
#            min_value = M[indx - 1, m - j - 1]
#
#        if (indx + 1) < n and min_value > M[indx + 1, m - j - 1]:
#
#            min_ind = indx + 1
#            min_value = M[indx + 1, m - j - 1]
#
#        camino.append(min_ind)
#
#    return M[min_ind, m-1], min_ind, camino

# Para eliminar los píxeles seleccionados en el camino de la costura simplemente
# desplazo la fila hacia arriba o la columna a la izquierda y elimino la última
# fila/columna
def removeSeam (image, camino):

    n, m = image.shape[:2]

#    if vertical:

    for i in range (0, n):

        for j in range (camino[i], m - 1):
            image[n - i - 1, j] = image[n - i - 1, j + 1]

    return np.delete(image, -1, 1)

    # Horizontal
#    for i in range (0, m):
#
#        for j in range (camino[i], n - 1):
#            image[j, m - i - 1] = image[j + 1, m - i - 1]
#
#    return np.delete(image, -1, 0)

# Para añadir píxeles a la imagen, hago los promedios con el vecino derecho y el
# vecino izquierdo (o el de arriba y abajo, si es una costura horizontal)
def addSeam (image, camino):

    n, m = image.shape[:2]

#    if vertical:

        # Nuevo tamaño de la imagen (Seguramente sea mejor añadirle una columna
        # de 0 que crear una nueva matriz)
        # Como hay que añadir píxeles, podría añadirsele una columna al final y
        # recorer desde el final, modificando solo los píxeles que hay que hay
        # que desplazar y modificar (la costura)
    img = np.empty((n, m + 1, 3), dtype=np.float)

    image = image.astype(np.float)

    for i in range (0, n):

        # La parte de la imagen que se mantiene intacta
        for j in range (0, camino[i]):
            img[n - i - 1, j] = image[n - i - 1, j]

        # Se comprueba si el píxel está en el borde, si está, no se hace el
        # promedio, se añade el nuevo pixel a la derecha, haciendo el promedio
        # con el vecino de la derecha
        if camino[i] > 0:
            left = image[n - i - 1, camino[i] - 1] * image[n - i - 1, camino[i] - 1]
            center = image[n - i - 1, camino[i]] * image[n - i - 1, camino[i]]

            new = (left + center)/2
            new = np.sqrt(new)

            img[n - i - 1, camino[i]] = new

        else:
            img[n - i - 1, camino[i]] = image[n - i - 1, camino[i]]

        # Si está en el borde derecho, se hizo el promedio con el vecino izquierdo
        # y se añadió el nuevo píxel. El pixel de la costura se mantiene igual
        if camino[i] < (m - 1):
            center = image[n - i - 1, camino[i]] * image[n - i - 1, camino[i]]
            right = image[n - i - 1, camino[i] + 1] * image[n - i - 1, camino[i] + 1]

            new = (center + right)/2
            new = np.sqrt(new)

            img[n - i - 1, camino[i] + 1] = new

        else:
            img[n - i - 1, camino[i] + 1] = image[n - i - 1, camino[i]]

        # Se copia el resto de la imagen
        for j in range (camino[i] + 1, m):
            img[n - i - 1, j + 1] = image[n - i - 1, j]

    img = img.astype(np.uint8)

    return img

    # Espejo de lo anterior para las costuras horizontales. Hay que reducirlo.
#    img = np.empty((n + 1, m, 3), dtype=np.float)
#
#    image = image.astype(np.float)
#
#    for i in range (0, m):
#
#        for j in range (0, camino[i]):
#            img[j, m - i - 1] = image[j, m - i - 1]
#
#        if camino[i] > 0:
#            up = image[camino[i] - 1, m - i - 1] * image[camino[i] - 1, m - i - 1]
#
#            center = image[camino[i], m - i - 1] * image[camino[i], m - i - 1]
#
#            new = (up + center)/2
#            new = np.sqrt(new)
#
#            img[camino[i], m - i - 1] = new
#
#        else:
#            img[camino[i], m - i - 1] = image[camino[i], m - i - 1]
#
#        if camino[i] < (n - 1):
#            center = image[camino[i], m - i - 1] * image[camino[i], m - i - 1]
#
#            down = image[camino[i] + 1, m - i - 1] * image[camino[i] + 1, m - i - 1]
#
#            new = (center + down)/2
#
#            new = np.sqrt(new)
#
#            img[camino[i] + 1, m - i - 1] = new
#
#        else:
#            img[camino[i] + 1, m - i - 1] = image[camino[i], m - i - 1]
#
#        for j in range (camino[i] + 1, n):
#            img[j + 1, m - i - 1] = image[j, m - i - 1]
#
#    img = img.astype(np.uint8)
#
#    return img

# Buscamos el orden en el que hay que aplicar las costuras para conseguir una
# imagen n x m -> n' x m' (fórmula 6 - página 5 del paper)
# Primero he supuesto que solo vamos a reducir imágenes, para ampliar, en vez
# de eliminar habría que duplicar los píxeles promediando con los vecinos que
# no estén en el camino de la costura
def seamsOrder (img, nn, nm):

    image = img.copy()

    n, m = image.shape[:2]

    r = n - nn + 1
    c = m - nm + 1

    T = np.zeros((r,c))

    options = np.zeros((r,c))


    options[0,0] = -1   # Si se quiere el mismo tamaño no necesitamos hacer ninguna costura
                        # 0 -> costura horizontal ; 1 -> costura vertical
    # Tenía problemas si solo eliminaba columnas, programé la solución más fácil
    # para comprobar si funcionaba
    # Rellenamos la primera columna de la tabla
    if r > 1:
        min_energy, indx, camino = horizontalSeam(np.rot90(image, k=-1, axes=(0, 1)))

        T[1,0] = T[0,0] + min_energy

        options[1,0] = 0

        image = removeSeam(np.rot90(image, k=-1, axes=(0, 1)), camino)

        hor_image = image.copy()

        for i in range (2, r):

            min_energy, indx, camino = horizontalSeam(hor_image)

            T[i,0] = T[i-1,0] + min_energy

            options[i,0] = 0

            hor_image = removeSeam(hor_image, camino, 0)

        vert_image = np.rot90(image, k=1, axes=(0, 1)).copy()

        for i in range (1, c):

            min_energy, indx, camino = verticalSeam(vert_image)

            T[0,i] = T[0,i-1] + min_energy

            options[0,i] = 1

            vert_image = removeSeam(vert_image, camino, 1)

        # No estoy segura si habría que ir modificando asi la imagen. Como se tiene
        # que rellenar la tabla para cada posible tamaño que puede tomar la imagen,
        # tendrá que, a la fuerza, eliminar c filas y r columnas, pero no se si tendría
        # que hacerse así
        for i in range (1, r):

            if c > 1:
                hor_min, hor_indx, path = horizontalSeam(np.rot90(image, k=-1, axes=(0, 1)))
                vert_min, vert_min, vert_path = verticalSeam(image)

                T[i,1] = min(T[i-1,1] + hor_min, T[i, 0] + vert_min)

                if T[i,1] == T[i, 0] + vert_min:
                    options[i,1] = 1

                vert_image = image.copy()

                for j in range (2, c-1):

                    hor_min, hor_indx, hor_path = horizontalSeam(np.rot90(vert_image, k=-1, axes=(0, 1)))
                    vert_min, vert_min, vert_path = verticalSeam(vert_image)

                    T[i,j] = min(T[i-1,j] + hor_min, T[i, j-1] + vert_min)

                    if T[i,j] == T[i, j-1] + vert_min:
                        options[i,j] = 1

                    vert_image = removeSeam(vert_image, vert_path, 1)

                image = removeSeam(np.rot90(image, k=-1, axes=(0, 1)), path, 0)

        print("Shape final: ", image.shape)
        return T, options

    # Si solo se quieren eliminar columnas
    # Espejo de lo anterior, adaptado para este caso. Lo mismo, manera rápida de
    # saber si funcionaba
    if c > 1:
        min_energy, indx, camino = verticalSeam(image)

        T[0,1] = T[0,0] + min_energy

        options[0,1] = 1

        image = removeSeam(image, camino, 1)

        vert_image = image.copy()

        for i in range (2, c):

            min_energy, indx, camino = verticalSeam(vert_image)

            T[0,i] = T[0,i-1] + min_energy

            options[0,i] = 1

            vert_image = removeSeam(vert_image, camino, 1)

        hor_image = image.copy()

        for i in range (1, r):

            min_energy, indx, camino = horizontalSeam(hor_image)

            T[i,0] = T[i-1,0] + min_energy

            options[i,0] = 1

            hor_image = removeSeam(hor_image, camino, 0)

        for j in range (1, c):

            if r > 1:
                hor_min, hor_indx, path = horizontalSeam(image)
                vert_min, vert_min, vert_path = verticalSeam(image)

                T[1,j] = min(T[1,j-1] + hor_min, T[0,j] + vert_min)

                if T[1,j] == T[0,j] + vert_min:
                    options[1,j] = 1

                hor_image = image.copy()

                for i in range (2, r-1):

                    hor_min, hor_indx, hor_path = horizontalSeam(hor_image)
                    vert_min, vert_min, vert_path = verticalSeam(hor_image)

                    T[i,j] = min(T[i-1,j] + hor_min, T[i, j-1] + vert_min)

                    if T[i,j] == T[i, j-1] + vert_min:
                        options[i,j] = 1

                    hor_image = removeSeam(hor_image, vert_path, 0)

                image = removeSeam(image, path, 1)

        print("Shape final: ", image.shape)
        return T, options

# Simplemente busca el orden en el que hay que eliminar las costuras
# Estuve fijandome que el valor que hay en la tabla de bits es el que se selecciona,
# Creo que sería tan facil con: si es un 1, le restamos a c y guardamos el 1.
# Si es un 0, le restamos a 0 y guardamos. Asi hasta que r y c sean 0
def selectSeamsOrder (image, T, options):

    r = T.shape[0] - 1
    c = T.shape[1] - 1
    cont = 1

    order = np.zeros((r+c))

    order[0] = options[r,c]

    r -= 1
    c -= 1

    while r > 0 and c > 0:

        if T[r, c-1] < T[r-1, c]:

            order[cont] = 1
            c -= 1

        else:
            r -= 1

        cont += 1

    while c > 0:

        order[cont] = 1
        c -= 1

        cont += 1

    while r > 0:

        r -= 1

        cont+= 1

    return order

def scaleAndRemoveSeams (img, nn, nm):

    n, m = img.shape[:2]
    scale_factor = max(nn/n, nm/m)
    height = int(n * scale_factor)
    width = int (m * scale_factor)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim)

    resized = np.rot90(resized, k=-1, axes=(0, 1))

    #Eliminamos las verticales o horizontales que sobren
    for i in range(height - nn):
        a, b, path = horizontalSeam(resized)
        resized = removeSeam (resized, path, 0)

    resized = np.rot90(resized, k=1, axes=(0, 1))

    for i in range(width - nm):
        a, b, path = verticalSeam(resized)
        resized = removeSeam(resized, path, 1)

    return resized

# Con el orden seleccionado, va eliminando horizontal o verticalmente las costuras
# de la imagen
def removeOrderSeams (img, order):

    image = img.copy()

    for o in order:

        if o:
            a, b, path = horizontalSeam (np.rot90(image, k=-1, axes=(0, 1)))
            image = removeSeam (np.rot90(image, k=-1, axes=(0, 1)), path, 0)

            image = np.rot90(image, k=1, axes=(0, 1))

        else:
            a, b, path = verticalSeam (image)
            image = removeSeam (image, path, 1)

    return image

# Similar a la anterior pero añadiendo
def addOrderSeams (img, order):

    image = img.copy()

    for o in order:

        if o:
            # Aqui (y similares) continuamente se calcula el camino. Habría que
            # pensar otra manera más eficiente. Por eso pensé en ir guardando los
            # camino que se generan al calcular la tabla T (como se hace con la
            # tabla de bits para horizontal y vertical) guardando el camino que
            # le correspondería, pero se guarda de 0 a r,c y se recorre de r,c a 0,
            # por lo que los píxeles no son los mismos.
            a, b, path = horizontalSeam (np.rot90(image, k=-1, axes=(0, 1)))
            image = addSeam (image, path, 0)

            image = np.rot90(image, k=1, axes=(0, 1))

        else:
            a, b, path = verticalSeam (image)
            image = addSeam (image, path, 1)

    return image

# Función para pintar las hebras seleccionadas (hay que tener en cuenta que a partir
# de la primera, los píxeles seleccionados pueden no corresponderse. Es una función
# sencilla para poder ver si funcionan los caminos. Si se quiere implementar bien,
# habría que tener en cuenta la mínima posición del píxel de cada fila/columa que
# se ha eliminado para poder adaptar a la hora de pintar. Asi como lo tengo se
# estaría pintando mal).
def drawSeams(vertical_seams, horizontal_seams, image):

    n, m = image.shape[:2]

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

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Prueba de funcionamiento
#image = readImage("surfista.jpeg", 1)
#
#img = image.copy()


#T, options = seamsOrder(image, image.shape[0], image.shape[1]-100)
#order = selectSeamsOrder (image, T, options)

#img1 = addOrderSeams (image, order)
#img2 = removeOrderSeams (image, order)

# Tarda muchisimo en ejecutar con esta imagen porque es grande.
# El resultado no es el que tiene que ser, falta refinamiento (es solo para ver
# que funciona el método)

#cv2.imshow("original", image)
#cv2.imshow("costuras eliminadas", img2)
#cv2.imshow("costuras añadidas", img1)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#
#plt.title("original")
#plt.imshow(img1)
#plt.show()
#
#plt.title("costuras elimindas")
#plt.imshow(img3)
#plt.show()
#
#plt.title("costuras")
#plt.imshow(img2)
#plt.show()
