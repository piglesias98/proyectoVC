# -*- coding: utf-8 -*-

import numpy as np

import Basics
import energias



def objectRemoval(image, remove_mask=None, preserve_mask=None, nn=0, nm=0):
    img = image.copy()
    n,m = image.shape[:2]
    if remove_mask.all() != None:
        nn, nm = Basics.maskSize(remove_mask)    
        
    if n-nn>m-nm:   #Eliminamos las filas 
        #Rotamos
#        img = np.rot90(img, k=-1, axes=(0, 1))
#        if remove_mask.all()!=None:
#            remove_mask = np.rot90(remove_mask, k=-1, axes=(0, 1))
#        if preserve_mask.all()!=None:
#            preserve_mask = np.rot90(preserve_mask, k=-1, axes=(0, 1))
        
        #Eliminamos las horizontales que sobren
        for i in range(abs(n - nn)):
            print("eliminamos", i)
            a, b, path = Basics.horizontalSeam(img, energias.forwardEnergy, remove_mask, preserve_mask)
            img = Basics.removeSeam(img, path)
            remove_mask = Basics.removeSeam(remove_mask, path)
            preserve_mask = Basics.removeSeam(preserve_mask, path)
            
        #Rotamos
#        return np.rot90(img, k=1, axes=(0, 1)
        return img
    else:
        #Eliminamos las verticales que sobren
        for i in range(abs(m - nm)):
            a, b, path = Basics.verticalSeam(img, energias.forwardEnergy, remove_mask, preserve_mask)
            img = Basics.removeSeam(img, path)
            remove_mask = Basics.removeSeam(remove_mask, path)
            preserve_mask = Basics.removeSeam(preserve_mask, path)
            
        return img
  