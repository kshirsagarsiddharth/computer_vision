#%%
# the previous technique didnot use any data agumentation 
# this training is slower than previous examples but lets use data agumentation for training 
# from keras import models 
from tensorflow.keras import models 
from tensorflow.keras import layers 
from tensorflow.keras.applications import VGG16
from tensorflow.python.ops.gen_math_ops import mod 

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
# %%
