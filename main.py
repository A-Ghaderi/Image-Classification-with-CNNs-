#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Import data and libraries ##
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[0])


# ## Preprocessing the data

# In[ ]:


## Normalizing the data ##
x_train.max()
x_train = x_train/255
x_test = x_test/255

x_train.shape

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

## One-hot encoded for categorical analysis ##
from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)


# ## Model building

# In[ ]:


## Model building ##
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

## Conv. layer ##
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
## Pooling layer ##
model.add(MaxPool2D(pool_size=(2, 2)))

## Flatten images from 28 by 28 to 764 before the final layer ##
model.add(Flatten())

## Dense hidden layer ##
model.add(Dense(128, activation='relu'))

## Output layer ##
model.add(Dense(10, activation='softmax'))

## Compiler ##
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary


# In[ ]:


model.fit(x_train,y_cat_train,epochs=10)


# ## Model evaluation

# In[ ]:


model.metrics_names
model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_test)
y_cat_test.shape

y_cat_test[0]
predictions[0]
y_test


# In[ ]:


print(classification_report(y_test,predictions))

