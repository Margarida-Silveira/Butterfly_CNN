from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
from PIL import Image




def transform(img, mask):
    random_transformation = np.random.randint(1,4)
    if random_transformation == 1:  # reverse first dimension
        img = img[::-1,:,:]
        mask = mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        img = img[:,::-1,:]
        mask = mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        img = img.transpose([1,0,2])
        mask = mask.transpose([1,0,2])
    else:
        pass
    
    return (img,mask)

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



import keras as keras
from keras.preprocessing import sequence
from keras.preprocessing import image
    

class TrainDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64,64), n_channels=8,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img = np.empty((self.batch_size, *self.dim, 3))
        mask = np.empty((self.batch_size, *self.dim, self.n_classes))
       
        msk = np.empty((*self.dim,self.n_classes))
#        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im=Image.open('images_rgb/' + ID)
            im=np.array(im)/255
            if  self.n_classes==3:
                a=Image.open('segmentation_gt_3/pupil/' + ID)
                a=np.array(a)
                msk[:,:,0]=a
                a=Image.open('segmentation_gt_3/ring/' + ID)
                a=np.array(a)
                msk[:,:,1]=a[:,:,1]/255
                msk[:,:,2]=np.logical_not(np.logical_or(msk[:,:,0],msk[:,:,1]))
            else:
                a=Image.open('segmentation_gt_2/' + ID)
                a=np.array(a)
                msk[:,:,0]=a[:,:,0]/255
                msk[:,:,1]=np.logical_not(msk[:,:,0])
            
            #im,msk=transform(im, msk)
           
            img[i,] = np.expand_dims(im,axis=0)
            mask[i,] = np.expand_dims(msk,axis=0)
          

        return img, mask

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64,64), n_channels=8,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img = np.empty((self.batch_size, *self.dim, 3))
        mask = np.empty((self.batch_size, *self.dim, self.n_classes))
        msk = np.empty((*self.dim, self.n_classes))
       
        # Generate data sem data augmentation
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im=Image.open('images_rgb/' + ID)
            im=np.array(im)/255
            #im=im.resize((64,64), Image.BILINEAR)
            if  self.n_classes==3:
                a=Image.open('segmentation_gt_3/pupil/' + ID)
                a=np.array(a)
                msk[:,:,0]=a
                a=Image.open('segmentation_gt_3/ring/' + ID)
                a=np.array(a)
                msk[:,:,1]=a[:,:,1]/255
                msk[:,:,2]=np.logical_not(np.logical_or(msk[:,:,0],msk[:,:,1]))
            else:
                a=Image.open('segmentation_gt_2/' + ID)
                a=np.array(a)
                msk[:,:,0]=a[:,:,0]/255
                msk[:,:,1]=np.logical_not(msk[:,:,0])            
                        
            img[i,] = np.expand_dims(im,axis=0)
            mask[i,] = np.expand_dims(msk,axis=0)

        return img, mask


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,img in enumerate(npyfile):
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        
