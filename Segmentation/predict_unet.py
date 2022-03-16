
from unet_model import *
import os
import numpy as np 
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


num_class=2
input_path='images_rgb/'
save_path='segmentation_results/'


names = os.listdir(input_path)
if num_class==3:
    model = unet_model(n_classes=num_class, class_weights=[25, 2, 1])
    model_name='unet_3classes_weighted.hdf5'   
else:
    model = unet_model(n_classes=num_class, class_weights=[1.5 ,1] )
    model_name='unet_2classes_weighted.hdf5'
   
model.load_weights(model_name)


for i in range(len(names)):
    img_name=names[i]
    im=np.array(Image.open(input_path + img_name))
    img= np.expand_dims(im,axis=0)/255
    ypred=np.squeeze(model.predict(img))
      
    if num_class==2:
        ypred=np.argmax(ypred,axis=2)*255
    
    io.imsave(os.path.join(save_path,"%d_predict.png"%i),ypred)
    	
    
