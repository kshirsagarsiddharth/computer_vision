#%%
# the previous technique didnot use any data agumentation 
# this training is slower than previous examples but lets use data agumentation for training 
# from keras import models 

#this technique is  feature extraction
from tensorflow.keras import models 
from tensorflow.keras import layers 
from tensorflow.keras.applications import VGG16
#from tensorflow.python.ops.gen_math_ops import mod 
import os
import numpy as np 
base_dir = r'D:\project\udemy_cv\Computer-Vision-with-Python\deep_learning\keras_basis.py\cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train') 
validation_dir = os.path.join(base_dir,'validation') 
test_dir = os.path.join(base_dir,'test')
conv_base = VGG16(weights='imagenet',
                  include_top=False, # wethr to include the densly connected network on top of the network 
                  input_shape=(150,150,3)
)

model = models.Sequential()  
model.add(conv_base) 
model.add(layers.Flatten()) 
model.add(layers.Dense(256, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid')) 
model.summary()

# before the model is compiled it is important to freeze the convolution base 
# Freezing a layer or set of layers means preventing their weights being updated during the 
# training process 
# if this is not done the representation previously learned from 
# convolution base are are modified during the training. 
# because the Dense layer on top are randomly inatilized very large 
# weights updates will propogate through the network 
# effectively destroying the representations previously learned 



# %%


print("Number of trainable weights before freezing the conv base:", len(model.trainable_weights))
conv_base.trainable = False 
print("Number of trainable weights after freezing the conv base:", len(model.trainable_weights))

# with the above setup only weights from the two dense layers will be trained 
# and these changes should be done before the model is compiled 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import optimizers 

train_datagen = ImageDataGenerator(
    rescale= 1./255, 
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range= 0.2, 
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator( 
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150,150),
            batch_size= 20,
            class_mode='binary'
)


validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'

)


model.compile(loss = 'binary_crossentropy',
optimizer= optimizers.RMSprop(learning_rate=2e-5),
metrics = ['accuracy']
)


# %%

history = model.fit(train_generator,
steps_per_epoch=100,
epochs = 30,
validation_data=validation_generator,
validation_steps = 50,
shuffle=True
)
# %%
history.history['accuracy']
# %%
history.history['val_accuracy']
# %%
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label = 'train_loss')
plt.plot(val_loss, label = 'val_loss')
plt.xlabel('no epochs')
plt.ylabel('loss')
plt.legend()
# %%
import matplotlib.pyplot as plt
loss = history.history['accuracy']
val_loss = history.history['val_accuracy']
plt.plot(loss, label = 'train_acc')
plt.plot(val_loss, label = 'val_acc')
plt.xlabel('no epochs')
plt.ylabel('acc')
plt.legend()
# %%
