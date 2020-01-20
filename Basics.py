# -*- coding: utf-8 -*-

import cv2
import numpy as np

import energias

# Leer la imagen de entrada
# Por defecto, las imagenes se leen a color
def readImage (filename, flagColor = 1):

    return cv2.imread(filename, flagColor)

def maskSize(mask):
    rows, cols = np.where(mask < 200)
    height = np.amax(rows) - np.amin(rows) + 1
    width = np.amax(cols) - np.amin(cols) + 1
    return height, width

def crearCamino (M):

    n, m = M.shape[:2]

    camino = np.zeros((m), dtype=np.int)

    camino[0] = np.argmin(M[-1])

    # Camino de la costura, buscando la menor energía posible
    for i in range (1, n):

        indy = camino[i-1]

        min_ind = indy
        min_value = M[n - i - 1, indy]

        if (indy - 1) > -1 and min_value > M[n - i - 1, indy - 1]:

            min_ind = indy - 1
            min_value = M[n - i - 1, indy - 1]

        if (indy + 1) < m and min_value > M[n - i - 1, indy + 1]:

            min_ind = indy + 1
            min_value = M[n - i - 1, indy + 1]

        camino[i] = min_ind

    return min_value, min_ind, camino

def Seam (image, energy):

    n, m = image.shape[:2]

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
def verticalSeam (image, funcion):

    energy = funcion(image)

    return (Seam(image, energy))

# Costura óptima vertical
def horizontalSeam (image, funcion):

    return verticalSeam(np.rot90(image, k=-1, axes=(0, 1)), funcion)

# Para eliminar los píxeles seleccionados en el camino de la costura simplemente
# desplazo la fila hacia arriba o la columna a la izquierda y elimino la última
# fila/columna
def removeSeam (image, camino):

    n, m = image.shape[:2]

    for i in range (0, n):

        for j in range (camino[i], m - 1):
            image[n - i - 1, j] = image[n - i - 1, j + 1]

    return np.delete(image, -1, 1)

# Para añadir píxeles a la imagen, hago los promedios con el vecino derecho y el
# vecino izquierdo (o el de arriba y abajo, si es una costura horizontal)
def addSeam (image, camino):

    n, m = image.shape[:2]

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

# Buscamos el orden en el que hay que aplicar las costuras para conseguir una
# imagen n x m -> n' x m' (fórmula 6 - página 5 del paper)
# Primero he supuesto que solo vamos a reducir imágenes, para ampliar, en vez
# de eliminar habría que duplicar los píxeles promediando con los vecinos que
# no estén en el camino de la costura

# HAY QUE COMPROBAR QUE SE QUIERA ELIMINAR AL MENOS UNA COLUMNA, SI NO, GIRAR LA IMAGEN
def seamsOrder (img, nn, nm, funcion=energias.forwardEnergy):

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

    min_energy, indx, camino = verticalSeam(image)

    T[0,1] = T[0,0] + min_energy

    options[0,1] = 1

    image = removeSeam(image, camino)

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

        hor_image = np.rot90(removeSeam(np.rot90(hor_image, k=-1, axes=(0, 1)), camino, 0), k=1, axes=(0,1))

    for j in range (1, c):

        if r > 1:
            hor_min, hor_indx, hor_path = horizontalSeam(image)
            vert_min, vert_min, path = verticalSeam(image)

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

                hor_image = np.rot90(removeSeam(np.rot90(hor_image, k=-1, axes=(0, 1)), hor_path, 0), k=1, axes=(0,1))

            image = removeSeam(image, path, 1)

    return T, options

# Simplemente busca el orden en el que hay que eliminar las costuras
# Estuve fijandome que el valor que hay en la tabla de bits es el que se selecciona,
# Creo que sería tan facil con: si es un 1, le restamos a c y guardamos el 1.
# Si es un 0, le restamos a r y guardamos 0. Asi hasta que r y c sean 0
def selectSeamsOrder (image, T, options):

    r = T.shape[0] - 1
    c = T.shape[1] - 1
    cont = 0

    order = np.zeros((r+c))

    while r > 0 or c > 0:

        if options[r,c]:
            order[cont] = 1
            c -= 1

        else:
            order[cont] = 0
            r -= 1

        if c < 0 or r < 0:
            print("C = %d, R = %d" % (c, r))

#    order[0] = options[r,c]
#
#    r -= 1
#    c -= 1
#
#    while r > 0 and c > 0:
#
#        if T[r, c-1] < T[r-1, c]:
#
#            order[cont] = 1
#            c -= 1
#
#        else:
#            r -= 1
#
#        cont += 1
#
#    while c > 0:
#
#        order[cont] = 1
#        c -= 1
#
#        cont += 1
#
#    while r > 0:
#
#        r -= 1
#
#        cont+= 1

    return order

def scaleAndCarve (img, nn, nm, accion=removeSeam, energia=energias.forwardEnergy):

    n, m = img.shape[:2]

    if accion == removeSeam: scale_factor = max(nn/n, nm/m)
    else: scale_factor = min(nn/n, nm/m)

    height = int(n * scale_factor)
    width = int (m * scale_factor)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim)

    resized = np.rot90(resized, k=-1, axes=(0, 1))

    #Eliminamos las verticales o horizontales que sobren
    for i in range(abs(height - nn)):
        a, b, path = verticalSeam(resized, energia)
        resized = accion(resized, path)

    resized = np.rot90(resized, k=1, axes=(0, 1))

    for i in range(abs(width - nm)):
        a, b, path = verticalSeam(resized, energia)
        resized = accion(resized, path)

    return resized

# Con el orden seleccionado, va eliminando horizontal o verticalmente las costuras
# de la imagen
def removeOrderSeams (img, order, funcion=energias.forwardEnergy):

    image = img.copy()

    for o in order:

        if o: image = np.rot90(image, k=-1, axes=(0, 1))

        a, b, path = verticalSeam (image)
        image = removeSeam (image, path, 1)

        if o: image = np.rot90(image, k=1, axes=(0, 1))

    return image


def carve (img, nn, nm, accion=removeOrderSeams, energia=energias.forwardEnergy):

    n, m = img.shape[:2]

    if (nm - m) == 0:
        img = np.rot90(img, k=-1, axes=(0, 1))

    T, options = seamsOrder (img, nn, nm, energia)

    order = selectSeamsOrder (img, T, options)

    return accion(img, order, funcion=energias.forwardEnergy)

# Similar a la anterior pero añadiendo
def addOrderSeams (img, order, funcion):

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

def removeEnergy(energy, mask):

    n, m = energy.shape[:2]

    for i in range (n):
        for j in range (m):

            if mask[i,j] < 255:
                energy[i,j] = -100

    return energy

def preserveEnergy(energy, mask):

    n, m = energy.shape[:2]

    maxi = energy.max() + 1000

    for i in range (n):
        for j in range (m):

            if mask[i,j] < 255:
                energy[i,j] = maxi

    return energy
