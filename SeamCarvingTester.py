# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""

import Basics
import SeamsCarving
import energias

'''
Leemos las imágenes de prueba
'''

#Imágenes para comprobación de energía
playa = Basics.readImage("imagenes/playa.jpg", 1)


#Imágenes para content aware resizing
campo = Basics.readImage("imagenes/campo.jpg", 1)

arbol1 = Basics.readImage("imagenes/arbol1.jpg", 1)

arco1 = Basics.readImage("imagenes/arco.jpg", 1)

arco2 = Basics.readImage("imagenes/arco2.jpg", 1)

hombre = Basics.readImage("imagenes/hombre.jpg", 1)

roca = Basics.readImage("imagenes/roca.jpg", 1)

banco = Basics.readImage("imagenes/banco.jpg", 1)



#Imágenes para Object Removal
arbol2 = Basics.readImage("imagenes/arbol2.jpg", 1)
arbol_n = Basics.readImage("imagenes/mascara_arbol.jpg", 0)

harry = Basics.readImage("imagenes/harry.jpg", 1)
harry_p = Basics.readImage("imagenes/positivo_harry.jpg", 0)
harry_n = Basics.readImage("imagenes/negativo_harry.jpg", 0)

coche = Basics.readImage("imagenes/coches.jpg", 1)
coche_n1 = Basics.readImage("imagenes/mascara_coches1.jpg", 0)
coche_n2 = Basics.readImage("imagenes/mascara_coches2.jpg", 0)


'''
Content Aware Resizing
'''

#print("----------------------ENERGÍAS------------------------------")
#print("Energía simple vs. Energía simple RGB")
#playa_simple_energy = energias.simpleEnergy(playa)
#playa_simple_energyRGB = energias.simpleEnergyRGB(playa)
#
#Basics.representar_imagenes([playa_simple_energy, playa_simple_energyRGB], ["Energía Simple", "Energía Simple RGB"])
#
#input("Pulse Enter para continuar")
#print("Reducción energía simple vs. energía simple RGB vs. forward energy")
#arco_simple_energy = SeamsCarving.contentAwareResizing (arco1, arco1.shape[0], arco1.shape[1] - 50, efficiency=False, energy = energias.simpleEnergy, draw=True)
#arco_simple_energy_RGB = SeamsCarving.contentAwareResizing (arco1, arco1.shape[0], arco1.shape[1] - 50, efficiency=False, energy = energias.simpleEnergyRGB, draw=True)
#arco_forward_energy = SeamsCarving.contentAwareResizing (arco1, arco1.shape[0], arco1.shape[1] - 50, efficiency=False, energy = energias.forwardEnergy, draw=True)
#
#Basics.representar_imagenes([arco_simple_energy[0], arco_simple_energy_RGB[0] ,arco_forward_energy[0],
#                             arco_simple_energy[1], arco_simple_energy_RGB[1], arco_forward_energy[1]],
#                            ["Simple Energy", "Simple Energy RGB", "Forward Energy",
#                             "Simple Energy", "Simple Energy RGB", "Forward Energy"], n_col=3)
#input("Pulse Enter para continuar")

#print("Reducción energía simple vs. energía simple RGB vs. forward energy")
#banco_simple_energy_RGB = SeamsCarving.contentAwareResizing (banco, banco.shape[0], banco.shape[1] - 130, efficiency=False, energy = energias.simpleEnergyRGB, draw=True)
#banco_forward_energy = SeamsCarving.contentAwareResizing (banco, banco.shape[0], banco.shape[1] - 130, efficiency=False, energy = energias.forwardEnergy, draw=True)
#
#Basics.representar_imagenes([banco_simple_energy_RGB[0] ,banco_forward_energy[0],
#                             banco_simple_energy_RGB[1], banco_forward_energy[1]],
#                            ["Simple Energy RGB", "Forward Energy", 
#                             "Simple Energy RGB", "Forward Energy"], n_col=2, tam = 50)


#input("Pulse Enter para continuar")

#print("-------------------CONTENT AWARE RESIZING-------------------")
#
#print("Disminución en filas y columnas")
#
#arbol1_red = SeamsCarving.contentAwareResizing (arbol1, arbol1.shape[0]-20, arbol1.shape[1] - 35, efficiency=False, energy = energias.forwardEnergy, draw=True)
#arbol1_red_eff = SeamsCarving.contentAwareResizing (arbol1, arbol1.shape[0]-20, arbol1.shape[1] - 35, efficiency=True, energy = energias.forwardEnergy, draw=True)
#tit1 = "Reducción: " + "Filas = " + str(arbol1_red[0].shape[0]) + "Columnas = " + str(arbol1_red[0].shape[1])
#tit1_eff = "Reducción eficiente: " + "Filas = " + str(arbol1_red_eff[0].shape[0]) + "Columnas = " + str(arbol1_red_eff[0].shape[1])
#
#Basics.representar_imagenes([arbol1_red[0] ,arbol1_red_eff[0],
#                             arbol1_red[1], arbol1_red_eff[1]],
#                            [tit1, tit1_eff, tit1, tit1_eff], n_col=2, tam = 50)

#input("Pulse Enter para continuar")
#print("Aumento en filas y columnas")
#
#campo_aum = SeamsCarving.contentAwareResizing (campo, campo.shape[0]+35, campo.shape[1]+20, efficiency=False, energy = energias.forwardEnergy, draw=True)
#campo_aum_eff = SeamsCarving.contentAwareResizing (campo, campo.shape[0]+35, campo.shape[1]+20, efficiency=True, energy = energias.forwardEnergy, draw=True)
#tit1 = "Aumento: " + "Filas = " + str(campo_aum[0].shape[0]) + ", Columnas = " + str(campo_aum[0].shape[1])
#tit1_eff = "Aumento eficiente: " + "Filas = " + str(campo_aum_eff[0].shape[0]) + ", Columnas = " + str(campo_aum_eff[0].shape[1])
#
#Basics.representar_imagenes([campo_aum[0] ,campo_aum_eff[0],
#                             campo_aum[1], campo_aum_eff[1]],
#                            [tit1, tit1_eff, tit1, tit1_eff], n_col=2, tam = 50)


#input("Pulse Enter para continuar")
#print("Aumento en filas y disminución en columnas")
#
#roca_ad = SeamsCarving.contentAwareResizing (roca, roca.shape[0]+30, roca.shape[1]-20, efficiency=False, energy = energias.forwardEnergy, draw=True)
#roca_ad_eff = SeamsCarving.contentAwareResizing (roca, roca.shape[0]+30, roca.shape[1]-20, efficiency=True, energy = energias.forwardEnergy, draw=True)
#tit1 = "Resize: " + "Filas = " + str(roca_ad[0].shape[0]) + "Columnas = " + str(roca_ad[0].shape[1])
#tit1_eff = "Resize eficiente: " + "Filas = " + str(roca_ad_eff[0].shape[0]) + "Columnas = " + str(roca_ad_eff[0].shape[1])
##
#Basics.representar_imagenes([roca_ad[0] ,roca_ad_eff[0],
#                             roca_ad[1], roca_ad_eff[1]],
#                            [tit1, tit1_eff, tit1, tit1_eff], n_col=2, tam = 50)

#input("Pulse Enter para continuar")
#print("Disminución en filas y aumento en columnas")
#
#hombre_da = SeamsCarving.contentAwareResizing (hombre, hombre.shape[0]-30, hombre.shape[1]+20, efficiency=False, energy = energias.forwardEnergy, draw=True)
#hombre_da_eff = SeamsCarving.contentAwareResizing (hombre, hombre.shape[0]-30, hombre.shape[1]+20, efficiency=True, energy = energias.forwardEnergy, draw=True)
#tit1 = "Resize: " + "Filas = " + str(hombre_da[0].shape[0]) + "Columnas = " + str(hombre_da[0].shape[1])
#tit1_eff = "Resize eficiente: " + "Filas = " + str(hombre_da_eff[0].shape[0]) + "Columnas = " + str(hombre_da_eff[0].shape[1])
##
#Basics.representar_imagenes([hombre_da[0] ,hombre_da_eff[0],
#                             hombre_da[1], hombre_da_eff[1]],
#                            [tit1, tit1_eff, tit1, tit1_eff], n_col=2, tam = 50)
#input("Pulse Enter para continuar")

#print("-------------------OBJECT REMOVAL-------------------")
#print("Eliminación con una máscara")
#eliminacion_arbol = SeamsCarving.objectRemoval(arbol2, remove_mask=arbol_n, rmask=True)
#Basics.representar_imagenes([arbol2, eliminacion_arbol], ["Original", "Eliminación"])

#input("Pulse Enter para continuar")
#eliminacion_coche1 = SeamsCarving.objectRemoval(coche, remove_mask=coche_n1, rmask=True)
#Basics.representar_imagenes([coche,eliminacion_coche1], ["Original", "Eliminación"])

#input("Pulse Enter para continuar")
#eliminacion_coche2 = SeamsCarving.objectRemoval(coche, remove_mask=coche_n2, rmask=True)
#Basics.representar_imagenes([coche,eliminacion_coche2], ["Original", "Eliminación"])

#input("Pulse Enter para continuar")
#print("Restaurado")
#
#restaurado = SeamsCarving.contentAwareResizing (eliminacion_coche2, coche.shape[0], coche.shape[1], efficiency=True, energy = energias.forwardEnergy, draw=False)
#
#Basics.representar_imagenes([coche,eliminacion_coche2, restaurado], ["Original", "Eliminación", "Restaurado"], n_col=3)

#input("Pulse Enter para continuar")
