# -*- coding: utf-8 -*-

import Basics

def contentAwareResizing (img, nn, nm, efficiency, energy):

    n, m = img.shape[:2]
    
    dif_n = n - nn
    dif_m = m - nm
    
    accion = [Basics.removeOrderSeams, Basics.addOrderSeams]
#    accion = [Basics.removeSeam, Basics.addSeam]
    funcion = Basics.scaleAndCarve

    if efficiency == False:
#        accion = [Basics.removeOrderSeams, Basics.addOrderSeams]
        funcion = Basics.carve

    if dif_n > 0:

        if dif_m < 0:
            img = funcion(img, nn, m, accion[0], energy)
            img = funcion(img, nn, nm, accion[1], energy)

        else:
            img = funcion(img, nn, nm, accion[0], energy)

    elif dif_n < 0:

        if dif_m > 0:
            img = funcion(img, n, nm, accion[0], energy)
            img = funcion(img, nn, nm, accion[1], energy)

        else:
            img = funcion(img, nn, m - abs(dif_m), accion[1], energy)

    else:

        if dif_m < 0:
            img = funcion(img, nn, nm, accion[1], energy)

        else:   
            img = funcion(img, n, nm, accion[0], energy)

    return img
