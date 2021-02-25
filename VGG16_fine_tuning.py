# it is only possible to fine tune the top layer of the convnet once the classifier is trained 

##########################################################################################
# steps for fine tuning 
# 1. Add your custom network on top of already trained base 
# 2. Freeze the base network 
# 3. Train the part you added 
# 4. Unfreeze some layers in the base network 
# 5. Jointly train both these layers and part you added 

# WHY first freeze the base network and train the dense network ?
## Because the Dense layer is randomly initialized and hence if some layers are unfreezed before the 
## Dense layer is trained the weights of conv base layer is affected 
########################################################################################
#%%
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

conv_base.summary()




# %%
"""
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
"""
conv_base.trainable = True 
# these conv layers should be trainable 
set_trainable = False 
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True 
    if set_trainable:
        layer.trainable = True 
    else:
        layer.trainable = False 

# In the above code we are iterating through all the layers of the convnet 
# and if we find block5_conv1 we switch the set_trainable = True 
# and once this is set to true all the layers will be set as trainable 
# %%


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
optimizer= optimizers.RMSprop(learning_rate=1e-5),
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