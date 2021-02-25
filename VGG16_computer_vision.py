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
##########################################################################
# There are two options
# OPTION ONE
#### 1. Running the convolution base over your dataset
#### 2. Recording its output to numpy array on disk.
#### 3. Using this data as input to standalone, densly connected classifier. 
#### 4. This is cheap to run but wont allow you to use data agumentation 

# OPTION TWO 
#### 1. Extending the model you have "conv_base" by adding the Dense Layer on top and running the whole thing end 
# to end on input data. 
#### 2. This allows to use data agumentation because every time a image goes through a convolution base it is seen by the model. 
#### 3. But this is expensive

# %%
import os  
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = r'D:\project\udemy_cv\Computer-Vision-with-Python\deep_learning\keras_basis.py\cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train') 
validation_dir = os.path.join(base_dir,'validation') 
test_dir = os.path.join(base_dir,'test')

datagen = ImageDataGenerator(rescale=1./255) 
batch_size = 20 

# %%

directory = train_dir 
# the directory
sample_count = 2000
# the samples stored in the directory 

# final feature map was 4,4,512 on this we will stick our dense layer 
features = np.zeros(shape=(sample_count,4,4,512)) 
labels = np.zeros(shape = (sample_count))

generator = datagen.flow_from_directory(
    directory,
    target_size=(150,150),
    batch_size=batch_size, # at a time take 20 images from the directory 
    class_mode='binary' 
)

# the generator DirectoryIterator yields tuples of (x,y) where x is a numpy array containing batch of 
# images with shape (batch_size,*target_size, channels) -- In this case --> (20,150,150,3) and y is label 

i = 0 

input_batch, labels_batch = next(generator)





# %%
# we put the image in pre trained convnet and extract the features
features_batch = conv_base.predict(input_batch)
i = 0
features[i * batch_size:(i + 1) * batch_size]  = features_batch 
labels[i * batch_size: (i + 1) * batch_size] = labels_batch
# %%
