import cv2
from keras.models import load_model
import numpy as np

model = load_model('models/model-013.h5')

labels_dict={0:'BARBIJO',1:'SIN BARBIJO'}
color_dict={0:(0,255,0),1:(0,0,255)}
cascada_rostro = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

img = cv2.imread('img6.jpg')
img=cv2.resize(img,(1000,1000))
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

coordenadas_rostros = cascada_rostro.detectMultiScale(img_gris, 1.3, 5)

for (x,y,ancho, alto) in coordenadas_rostros:
    face_img=img_gris[y:y+ancho,x:x+ancho]
    resized=cv2.resize(face_img,(100,100))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,100,100,1))
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    cv2.rectangle(img,(x,y),(x+ancho,y+alto),color_dict[label],2)
    cv2.rectangle(img,(x,y-40),(x+ancho,y),color_dict[label],-1)
    cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
 

cv2.imshow('Output', img)
print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
cv2.waitKey(0)
cv2.destroyAllWindows()