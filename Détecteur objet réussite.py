# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:36:03 2021

@author: Paul
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
# UTILISATION DU pack Yolo ssd mobilenet v3 large coco (2020)

thres = 0.5 # On impose un seuil pour détecter l'objet
nms_threshold = 0.2 #fonction pour ajuster les bords de la détection (compris de 0.1 à 1)  1 veut dire suppression, et 0.1 veut dire une suppression max 
cap = cv2.VideoCapture('video 2 s.mp4') # source vidéo : choisi pour un fichier .mp4 ou alors mettre #'1' pour la webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) # dimension pour la largeur 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #dimension pour la hauteur
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #valeur pour la luminosité 

classNames = [] # initialisation : liste vide 
with open('coco.names.txt','r') as f:  #ouvrir la liste contenant des objets que l'IA a été entraîner à assimiler
    classNames = f.read().splitlines() # définir une class qui prend en compte les éléments de la liste
print(classNames) # lecture puis afficher liste des objets

font = cv2.FONT_HERSHEY_PLAIN #détails
#font = cv2.FONT_HERSHEY_COMPLEX pour autres 
Colors = np.random.uniform(0, 255, size=(len(classNames), 3)) # définir pour chaque élément de la liste objet une couleur aléatoire

weightsPath = "frozen_inference_graph.pb" # Utilisation du cadre tensorflow
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt" #  On va charger le modèle pré- entraîné et exécuter un DNN est très simple dans OpenCV
net = cv2.dnn_DetectionModel(weightsPath,configPath) # on charge le modèle pré -entraîné
net.setInputSize(320,320) #  création d'un blob à partir de notre vidéo d' entrée (frame par frame) blob 320 x 320 
net.setInputScale(1.0/ 127.5) # pour l'échelle
net.setInputMean((127.5, 127.5, 127.5)) # définir la valeur moyenne du cadre : Paramètres: mean - Scalaire avec des valeurs moyennes qui sont soustraites des canaux. Retour: généré automatiquement
net.setInputSwapRB(True) # Indicateur qui indique que l'échange des premier et dernier canaux pour le cadre

while True:  # Quand on trouve un potentiel objet de la liste : On  ajoute un curseur pour sélectionner le niveau de confiance BoundingBox de 0 à 1. 
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold) #  on charge le modèle pré - entraîné avcc les valeurs de confiance pré supposé
    if len(classIds) != 0:  #si la liste qui contient l'identité de l'image identifié n'est pas vide
        for i in indices: # on va poursuivre par itération une procédure de vérification de la confiance
            i = i[0] # Maintenant que nous avons notre détection effectuée dans la variable de sortie, nous pouvons analyser simplement par une boucle for  
            box = bbox[i]
            confidence = str(round(confs[i],2))
            color = Colors[classIds[i][0]-1]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2) # épaisseur 
            cv2.putText(img, classNames[classIds[i][0]-1]+" "+confidence,(x+10,y+20), #confiance
                        font,1,color,2)
#             cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),
#                         font,1,colors[classId-1],2)

    cv2.imshow("Output",img)
    cv2.waitKey(1)