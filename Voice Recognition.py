

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


df=pd.read_csv("voice.csv")

df.head(10)

df.describe()

df.info()

y=pd.get_dummies(df.label)

x=df.drop(["label"],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

#Define Keras Model
model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Dense(15, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile the keras model
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# fit the keras model on the dataset
fitness=model.fit(X_train, Y_train, epochs=100 ,validation_data=(X_test, Y_test),shuffle=True)

model.summary()

plt.plot(range(len(fitness.history['accuracy'])),fitness.history['accuracy'],c='m')
plt.plot(range(len(fitness.history['accuracy'])),fitness.history['val_accuracy'],c='c')
plt.title("accuracy")
plt.legend(["accuracy","val_accuracy"])
plt.show()

plt.plot(range(len(fitness.history['loss'])),fitness.history['loss'],c='r')
plt.plot(range(len(fitness.history['loss'])),fitness.history['val_loss'],c='b')
plt.legend(['loss','val_loss'])
plt.title("loss")
plt.show()

...
# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Define Keras Model
#afzayesh teda noron ha

model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile the keras model
#optimizer ra mitavan SGD,adam,RMSprop,... ham gharar dad
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
fitness=model.fit(X_train, Y_train, epochs=100 ,validation_data=(X_test, Y_test),shuffle=True)

model.summary()

plt.plot(range(len(fitness.history['accuracy'])),fitness.history['accuracy'],c='m')
plt.plot(range(len(fitness.history['accuracy'])),fitness.history['val_accuracy'],c='c')
plt.title("accuracy")
plt.legend(["accuracy","val_accuracy"])
plt.show()

plt.plot(range(len(fitness.history['loss'])),fitness.history['loss'],c='r')
plt.plot(range(len(fitness.history['loss'])),fitness.history['val_loss'],c='b')
plt.legend(['loss','val_loss'])
plt.title("loss")
plt.show()

# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Define Keras Model

model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# model = Sequential()
# model.add(Dense(20, input_dim=20))
# model.add(Dense(15, activation='sigmoid'))
# model.add(Dense(2, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
fitness=model.fit(X_train, Y_train, epochs=100 ,validation_data=(X_test, Y_test),shuffle=True)

model.summary()

plt.plot(range(len(fitness.history['accuracy'])),fitness.history['accuracy'],c='m')
plt.plot(range(len(fitness.history['accuracy'])),fitness.history['val_accuracy'],c='c')
plt.title("accuracy")
plt.legend(["accuracy","val_accuracy"])
plt.show()

plt.plot(range(len(fitness.history['loss'])),fitness.history['loss'],c='r')
plt.plot(range(len(fitness.history['loss'])),fitness.history['val_loss'],c='b')
plt.legend(['loss','val_loss'])
plt.title("loss")
plt.show()

# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Define Keras Model

model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Dense(15, activation='tanh'))
model.add(Dense(2, activation='softmax'))

# model = Sequential()
# model.add(Dense(20, input_dim=20))
# model.add(Dense(15, activation='tanh'))
# model.add(Dense(2, activation='tanh'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
fitness=model.fit(X_train, Y_train, epochs=100 ,validation_data=(X_test, Y_test),shuffle=True)

model.summary()

plt.plot(range(len(fitness.history['accuracy'])),fitness.history['accuracy'],c='m')
plt.plot(range(len(fitness.history['accuracy'])),fitness.history['val_accuracy'],c='c')
plt.title("accuracy")
plt.legend(["accuracy","val_accuracy"])
plt.show()

plt.plot(range(len(fitness.history['loss'])),fitness.history['loss'],c='r')
plt.plot(range(len(fitness.history['loss'])),fitness.history['val_loss'],c='b')
plt.legend(['loss','val_loss'])
plt.title("loss")
plt.show()

# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

droup_out=[0.2,0.5,0.7]

for d in droup_out:
  model = Sequential()
  model.add(Dense(20, input_dim=20))
  model.add(Dropout(d))
  model.add(Dense(15, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  fitness=model.fit(X_train, Y_train, epochs=100 ,validation_data=(X_test, Y_test),shuffle=True)
  model.summary()
  
  plt.plot(range(len(fitness.history['accuracy'])),fitness.history['accuracy'],c='m')
  plt.plot(range(len(fitness.history['accuracy'])),fitness.history['val_accuracy'],c='c')
  plt.title("accuracy")
  plt.legend(["accuracy","val_accuracy"])
  plt.show()

  plt.plot(range(len(fitness.history['loss'])),fitness.history['loss'],c='r')
  plt.plot(range(len(fitness.history['loss'])),fitness.history['val_loss'],c='b')
  plt.legend(['loss','val_loss'])
  plt.title("loss ")
  plt.show()

  _, accuracy = model.evaluate(X_test, Y_test)
  print('Accuracy: %.2f' % (accuracy*100))
  print('-------------------------------------------------')


