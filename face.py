import deepface
from deepface import DeepFace
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

# package dependencies
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DeepID,
    DlibWrapper,
    ArcFace,
    SFace,
)
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

embedding1=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images.jpg',model_name='Facenet')
embedding4=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-4.jpg',model_name='Facenet')
embedding5=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-5.jpg',model_name='Facenet')
embedding6=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-6.jpg',model_name='Facenet')
embedding7=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-7.jpg',model_name='Facenet')
embedding8=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-8.jpg',model_name='Facenet')
embedding9=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-9.jpg',model_name='Facenet')
embedding10=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-10.jpg',model_name='Facenet')
embedding11=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-11.jpg',model_name='Facenet')
def llist(name):

 a=name[0].values()
 k=[]
 for i in a:
    k.append(i)

 return(k[0])

df = pd.DataFrame(np.array([llist(embedding11), llist(embedding10),llist(embedding9),llist(embedding8),llist(embedding7),llist(embedding6),llist(embedding5),llist(embedding4),llist(embedding1)]))

a=['ar','ar','ar','ar','ar','ar','ar','ar','ar']

df['name']=a


x=df.drop('name',axis=1)
y=df['name']
print(x)
print(y)

embedding12=DeepFace.represent(img_path=r'C:\Users\Ariya Rayaneh\Desktop\aron\images-12.jpg',model_name='Facenet')

model=KNeighborsClassifier()
model.fit(x,y)
y_pred=model.predict([llist(embedding12)])

print(y_pred)