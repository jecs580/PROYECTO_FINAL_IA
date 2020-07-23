from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np

# Carga de la data preprocesada
data=np.load('data.npy')
target=np.load('target.npy')

# Definiendo modelo
model=Sequential()

# Primera Capa 
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
# Reduccion de datos
model.add(MaxPooling2D(pool_size=(2,2)))

# Segunda Capa
model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Aplanamos la matriz
model.add(Flatten())
# Apagamos el 50% de las  Neuronales durante el entrenamiento, para evitar el sobre ajuste
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))

# conviertimos los números resultantes en probabilidades que suman uno
model.add(Dense(2,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Separando la data en entrenamiento y prueba
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

# Entrenamiento del Modelo
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)


# Evaluando la precision del modelo
print(model.evaluate(test_data,test_target))