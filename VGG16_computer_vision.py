#%%
from tensorflow.keras.applications import VGG16 
import numpy as np 

conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150,150,3))

# Three arguments are passed to the constructor 
# weights: specifies the weights checkpoint to initialize the model 
# include_top: refers to including the densly connected network on top of the network. 
# by default the densly connected network corresponds to 1000 classes of ImageNet. 
# we intend to use our own classifier dont need to include it. 

conv_base.summary()
# %%
