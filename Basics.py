# -*- coding: utf-8 -*-

import cv2
import numpy as np


from matplotlib import pyplot as plt
import energias

'''
Lectura de imágenes

Entrada:
    -> filename: camino relativo a la imagen
    -> flagColor: 1 -> Se lee a color ; 0 -> Se lee en escala de grises

Salida:
    -> imagen como matriz de uint8
'''
def readImage (filename, flagColor = 1):

    return cv2.imread(filename, flagColor)


'''
Representación de imágenes

Entrada:
    -> lista_imagen_leida: lista de imágenes que se quiere mostrar en pantalla
    -> lista_titulos: lista de títulos para las imágenes anteriores
    -> n_col: número de columnas en las que se van a representar las imágenes
    -> tam: tamaño que tendrán las imágenes

PRE:
    -> El tamaño de la lista lista_imagen_leida y la de lista_titulos debe ser
       el mismo
'''
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

'''
Número de filas y columnas que hay que eliminar para conseguir eliminar un objeto
de la imagen, a partir de la información de su máscaras

Entrada:
    -> mask: máscara que indica que parte de la imagen se quiere terminar.

Salida:
    -> height: número de filas que se van a quitar
    -> width: número de columnas que se van a quitar

PRE: la máscara deber haberse leido en monobanda
'''
def maskSize(mask):

    rows, cols = np.where(mask < 200)
    height = np.amax(rows) - np.amin(rows) + 1
    width = np.amax(cols) - np.amin(cols) + 1

    return height, width

'''
Crea el camino de seamns en función de las energías

Entrada:
    -> M: matriz de energía acumulativa

Salida:
    -> min_value: mínimo valor final de energía
    -> min_ind: índice del min_value
    -> camino: camino de píxeles que se deben eliminar (seam)
'''
def crearCamino (M):

    n, m = M.shape[:2]

    camino = np.zeros((n), dtype=np.int)

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

'''
Devuelve el camino creado por la matriz M

Entrada:
    -> image: imagen
    -> energy: matriz de energía

Salida:
    -> salida de la función "crearCamino"
'''
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

'''
Costura óptima vertical

Entrada:
    -> image: imagen
    -> funcion: función que calcula la energía
    -> remove_mask: máscara que se multiplica con energía para eliminar un objeto
    -> preserve_mask: máscara que se multiplica con la energía para conservar un objeto
    -> rmask: False -> no se usa el parámetro remove_mask
              True -> se usa el parámetro remove_mask
    -> pmask: False -> no se usa el parámetro remove_mask
              True -> se usa el parámetro remove_mask
Salida:
    -> salida de la función "Seam"
'''
def verticalSeam (image, funcion, remove_mask=None, preserve_mask=None, rmask=False, pmask=False):

    energy = funcion(image)

    if rmask:
        energy= removeEnergy(energy, remove_mask)

    if pmask:
        energy= preserveEnergy(energy, preserve_mask)

    return Seam(image, energy)

'''
Costura óptima horizontal

Entrada:
    -> image: imagen
    -> funcion: función que calcula la energía
    -> remove_mask: máscara que se multiplica con energía para eliminar un objeto
    -> preserve_mask: máscara que se multiplica con la energía para conservar un objeto
    -> rmask: False -> no se usa el parámetro remove_mask
              True -> se usa el parámetro remove_mask
    -> pmask: False -> no se usa el parámetro remove_mask
              True -> se usa el parámetro remove_mask
Salida:
    -> salida de la función "verticalSeam"
'''
def horizontalSeam (image, funcion, remove_mask=None, preserve_mask=None, rmask=False, pmask=False):

    if rmask:
         remove_mask = np.rot90(remove_mask, k=-1, axes=(0, 1))

    if pmask:
        preserve_mask = np.rot90(preserve_mask, k=-1, axes=(0, 1))

    return verticalSeam(np.rot90(image, k=-1, axes=(0, 1)), funcion, remove_mask, preserve_mask, rmask, pmask)

'''
Elimina una seam

Para eliminar los píxeles seleccionados en el camino de la costura simplemente
se desplaza la columna hacia la izquierda y se elimina la última columna de la
imagen

Entrada:
    -> image: imagen
    -> camino: seam a eliminar

Salida:
    -> imagen con seam eliminada
'''
def removeSeam (image, camino):

    n, m = image.shape[:2]

    for i in range (0, n):

        for j in range (camino[i], m - 1):
            image[n - i - 1, j] = image[n - i - 1, j + 1]

    return np.delete(image, -1, 1)

'''
Añade una seam

Para añadir píxeles a la imagen, se hace los promedios con el vecino derecho y el
vecino izquierdo

Entrada:
    -> image: imagen
    -> camino: seam a añadir

Salida:
    -> imagen con seam duplicada
'''
def addSeam (image, camino):

    n, m = image.shape[:2]

    # Nuevo tamaño de la imagen (Seguramente sea mejor añadirle una columna
    # de 0 que crear una nueva matriz)
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

'''
Elimina las seams en el orden seleccionado

Con el orden seleccionado, va eliminando horizontal o verticalmente las costuras
de la imagen

Entrada:
    -> img: imagen
    -> order: lista de 0's y 1's que indica si se tiene que eliminar una seam
              vertical u horizontal (0 -> horizontal ; 1 -> vertical)
    -> funcion: funcion para calcular la energía

Salida:
    -> imagen con todas las seams eliminadas
'''
def removeOrderSeams (img, order, funcion=energias.forwardEnergy, draw=False):

    image = img.copy()

    if draw:
        resultado = img.copy()

    for o in order:

        if o == 0:
            image = np.rot90(image, k=-1, axes=(0, 1))

            if draw:
                resultado = np.rot90(resultado, k=-1, axes=(0, 1))

        a, b, path = verticalSeam (image, funcion)
        image = removeSeam (image, path)

        if draw:
            resultado = drawSeams([path], [], resultado)

        if o == 0:
            image = np.rot90(image, k=1, axes=(0, 1))

            if draw:
                resultado = np.rot90(resultado, k=1, axes=(0, 1))

    if draw:
        return image, resultado
    
    return image

'''
Añade las semas en el orden seleccionado

Función para añadir que seams se van a duplicar. Se van acumulando los caminos a
eliminar hasta que se cambia de eje.

Entrada:
    -> img: imagen
    -> order: lista de 0's y 1's que indica si se tiene que añadir una seam
              vertical u horizontal (0 -> horizontal ; 1 -> vertical)
    -> funcion: funcion para calcular la energía

Salida:
    -> imagen con todas las seams duplicada
'''
def addOrderSeams (img, order, funcion=energias.forwardEnergy, draw=False):

    image = img.copy()

    if draw:
        resultado = img.copy()

    aux = img.copy()
    caminos = []
    orden = []

    anterior = order[0]

    for o in order:

        if o != anterior:

            for i in range (len(caminos)):

                if orden[i] == 0:
                    image = np.rot90(image, k=-1, axes=(0, 1))

                image = addSeam (image, caminos[i])

                if orden[i] == 0:
                    image = np.rot90(image, k=1, axes=(0, 1))

            if draw:
                if anterior:
                    resultado = drawSeams(caminos,[],resultado)

                else:
                    resultado = drawSeams([], caminos, resultado)

            orden = []
            caminos = []

            anterior = o

            aux = image.copy()

        orden.append(o)

        if o == 0:
            aux = np.rot90(aux, k=-1, axes=(0, 1))

        a, b, path = verticalSeam (aux, funcion)
        aux = removeSeam (aux, path)

        if o == 0:
            aux = np.rot90(aux, k=1, axes=(0, 1))

        caminos.append(path)

    for i in range (len(caminos)):

        if draw:
            if anterior:
                resultado = drawSeams(caminos,[],resultado)

            else:
                resultado = drawSeams([], caminos, resultado)

        if orden[i] == 0:
            image = np.rot90(image, k=-1, axes=(0, 1))

        image = addSeam (image, caminos[i])

        if orden[i] == 0:
            image = np.rot90(image, k=1, axes=(0, 1))

    if draw:
        return image, resultado

    return image

'''
Calcula el orden para añadir/eliminar seams

Buscamos el orden en el que hay que aplicar las costuras para conseguir una
imagen n x m -> n' x m' (fórmula 6 - página 5 del paper)

Entrada:
    -> image: imagen
    -> nn: nuevo tamaño de filas
    -> nm: nuevo tamaño de columnas
    -> funcion: función de energía que se utiliza (por defecto forwardEnergy)

Salida:
    -> T: matriz de energías, eligiendo siempre la mínima de las opciones
    -> options: matriz que guarda la opción (seam vertical u horizontal) que se
                ha elegido como mínima para cada casilla de T

PRE: Se tiene que poder eliminar, al menos, una columna (aunque no se quiera
     eliminar ninguna fila)
'''
def seamsOrder (img, nn, nm, funcion=energias.forwardEnergy):

    image = img.copy()

    n, m = image.shape[:2]

    r = abs(n - nn) + 1
    c = abs(m - nm) + 1

    T = np.zeros((r,c))

    options = np.zeros((r,c))

    options[0,0] = -1   # Si se quiere el mismo tamaño no necesitamos hacer ninguna costura
                        # 0 -> costura horizontal ; 1 -> costura vertical
    # Tenía problemas si solo eliminaba columnas, programé la solución más fácil
    # para comprobar si funcionaba
    # Rellenamos la primera columna de la tabla

    min_energy, indx, camino = verticalSeam(image, funcion)

    T[0,1] = T[0,0] + min_energy

    options[0,1] = 1

    image = removeSeam(image, camino)

    vert_image = image.copy()

    for i in range (2, c):

        min_energy, indx, camino = verticalSeam(vert_image, funcion)

        T[0,i] = T[0,i-1] + min_energy

        options[0,i] = 1

        vert_image = removeSeam(vert_image, camino)

#    hor_image = image.copy()
    hor_image = np.rot90(image, k=-1, axes=(0, 1))

    for i in range (1, r):

#        min_energy, indx, camino = horizontalSeam(hor_image, funcion)
        min_energy, indx, camino = verticalSeam(hor_image, funcion)

        T[i,0] = T[i-1,0] + min_energy

        options[i,0] = 1

#        hor_image = np.rot90(removeSeam(np.rot90(hor_image, k=-1, axes=(0, 1)), camino), k=1, axes=(0,1))
        hor_image = removeSeam(hor_image, camino)

    for j in range (1, c):

        if r > 1:
            hor_min, hor_indx, hor_path = horizontalSeam(image, funcion)
            vert_min, vert_min, path = verticalSeam(image, funcion)

            T[1,j] = min(T[1,j-1] + hor_min, T[0,j] + vert_min)

            if T[1,j] == T[0,j] + vert_min:
                options[1,j] = 1

#            hor_image = image.copy()
            hor_image = np.rot90(image, k=-1, axes=(0, 1))

            for i in range (2, r-1):

                hor_min, hor_indx, hor_path = verticalSeam(hor_image, funcion)
                vert_min, vert_min, vert_path = verticalSeam(np.rot90(hor_image, k=1, axes=(0, 1)), funcion)

                T[i,j] = min(T[i-1,j] + hor_min, T[i, j-1] + vert_min)

                if T[i,j] == T[i, j-1] + vert_min:
                    options[i,j] = 1

                hor_image = removeSeam(hor_image, hor_path)

            image = removeSeam(image, path)

    return T, options

'''
Busca el orden óptimo para eliminar las seams

Entrada:
    -> image: imagen
    -> T: matriz de energías para cada tamaño posible
    -> options: matriz que guarda la opción (seam vertical u horizontal) que se
                ha elegido como mínima para cada casilla de T

Salida:
    -> Orden en el que se tienen que eliminar las seams
'''
def selectSeamsOrder (image, T, options):

    r = T.shape[0] - 1
    c = T.shape[1] - 1
    cont = 0

    order = np.zeros((r+c))

    while cont < order.shape[0]:

        if c < 0 or r < 0:
            print("ERROR: C = %d, R = %d" % (c, r))

        if options[r,c]:
            if c > -1:
                order[cont] = 1
                c -= 1

            else:
                order[cont] = 0
                r -= 1

        else:
            if r > -1:
                order[cont] = 0
                r -= 1
            else:
                order[cont] = 1
                c -= 1

        cont += 1

    return order

'''
Modifica el tamaño de la imagen de manera no eficiente

Entrada:
    -> img: imagen
    -> nn: nuevo número de filas
    -> nm: nuevo número de columnas
    -> accion: indica si se va a eliminar o añadir seams
    -> energia: función de energía que se va a utilizar

Salida:
    -> imagen con las nuevas dimensiones especificadas
'''
def carve (img, nn, nm, accion=removeOrderSeams, energia=energias.forwardEnergy, draw=False):

    n, m = img.shape[:2]

    r = abs(n - nn)
    c = abs(m - nm)
    girar = c == 0

    if girar:
        img = np.rot90(img, k=-1, axes=(0, 1))

    if girar or (r == 0):
        order = np.ones((r+c))

    else:
        T, options = seamsOrder (img, nn, nm, energia)

        order = selectSeamsOrder (img, T, options)

    image = accion(img, order, energia, draw)

    if girar:
        return np.rot90(image, k=1, axes=(0, 1))

    return image

'''
Modifica el tamaño de la imagen de manera eficiente

Primero escala la imagen y después quita o añade las seams necesarias para conseguir
el tamaño especificado

Entrada:
    -> img: imagen
    -> nn: nuevo número de filas
    -> nm: nuevo número de columnas
    -> accion: indica si se va a eliminar o añadir seams
    -> energia: función de energía que se va a utilizar

Salida:
    -> imagen con las nuevas dimensiones especificadas
'''
def scaleAndCarve (img, nn, nm, accion=removeOrderSeams, energia=energias.forwardEnergy, draw=False):

    n, m = img.shape[:2]
    print("n", n, "m", m)
    if accion == removeOrderSeams: scale_factor = max(nn/n, nm/m)
    else: scale_factor = min(nn/n, nm/m)

    height = int(n * scale_factor)
    width = int (m * scale_factor)
    dim = (width,height)
    print("dim", dim)
    # resize image
    resized = cv2.resize(img, dim)
    print("resize shape", resized.shape)
    #Rotamos
    if abs(height - nn) != 0:
        print("height - nn", height-nn)
        order = np.ones((abs(height - nn)))

        resized = np.rot90(resized, k=-1, axes=(0, 1))

    elif abs(width - nm) != 0:
        order = np.ones((abs(width - nm)))

    #Eliminamos las verticales o horizontales que sobren
    resized = accion(resized, order, energia, draw)

#    for i in range(abs(height - nn)):
#        a, b, path = verticalSeam(resized, energia)
#        resized = accion(resized, path)
#
#    for i in range(abs(width - nm)):
#        a, b, path = verticalSeam(resized, energia)
#        resized = accion(resized, path)

    if abs(height - nn) != 0:
        resized = np.rot90(resized, k=1, axes=(0, 1))


#    resized = np.rot90(resized, k=-1, axes=(0, 1))
#    if remove_mask.all()!=None:
#        remove_mask = np.rot90(remove_mask, k=-1, axes=(0, 1))
#    if preserve_mask.all()!=None:
#        preserve_mask = np.rot90(preserve_mask, k=-1, axes=(0, 1))
#
#
#    #Eliminamos las verticales o horizontales que sobren
#    for i in range(abs(height - nn)):
#        a, b, path = verticalSeam(resized, energia, remove_mask, preserve_mask)
#        resized = accion(resized, path)
#        remove_mask = accion(remove_mask, path)
#        preserve_mask = accion(preserve_mask, path)
#
#    resized = np.rot90(resized, k=1, axes=(0, 1))
#    if remove_mask.all()!=None:
#        remove_mask = np.rot90(remove_mask, k=1, axes=(0, 1))
#    if preserve_mask.all()!=None:
#        preserve_mask = np.rot90(preserve_mask, k=1, axes=(0, 1))
#
#    for i in range(abs(width - nm)):
#        a, b, path = verticalSeam(resized, energia, remove_mask, preserve_mask)
#        resized = accion(resized, path)
#        remove_mask = accion(remove_mask, path)
#        preserve_mask = accion(preserve_mask, path)

    return resized

'''
Modifica la matriz de energía para adaptarse a la máscara de la imagen

Esta función es para modificar la energía con una máscara para eliminar un objeto

Entrada:
    -> energy: matriz de energía
    -> mask: máscara de la imagen

Salida:
    -> matriz de energía de la imagen modificada por la máscara
'''
def removeEnergy(energy, mask):

    n, m = energy.shape[:2]

    for i in range (n):
        for j in range (m):

            if mask[i,j] < 255:
                energy[i,j] = -100

    return energy

'''
Modifica la matriz de energía para adaptarse a la máscara de la imagen

Esta función es para modificar la energía con una máscara para conservar un objeto

Entrada:
    -> energy: matriz de energía
    -> mask: máscara de la imagen

Salida:
    -> matriz de energía de la imagen modificada por la máscara
'''
def preserveEnergy(energy, mask):

    n, m = energy.shape[:2]

    maxi = energy.max() + 1000

    for i in range (n):
        for j in range (m):

            if mask[i,j] < 255:
                energy[i,j] = maxi

    return energy

'''
Función para pintar las seams dn la imagen

Entrada:
    -> vertical_seams: seams verticales que se quieren pintar
    -> horizontal_seams: seams horizontales que se quieren pintar
    -> image: imagen

Salida:
    -> imagen con las seams indicadas en rojo
'''
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
