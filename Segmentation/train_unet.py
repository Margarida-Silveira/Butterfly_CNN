
from unet_model import *
from data2 import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from skimage import segmentation
import random

    

import json

with open('unet_partition.json', 'r') as json_file:
      partition=json.load(json_file)            

train_samples=len(partition['train'])
val_samples=len(partition['validation'])


num_class=3
batch_size=5
steps_per_epoch=train_samples//batch_size
epochs=50
params = {'dim': (128,128),          
           'batch_size': batch_size,
          'n_classes':num_class,
          'n_channels': 3, #RGB images
          'shuffle': True}


training_generator = TrainDataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)


if num_class==3:
    model = unet_model(n_classes=num_class, class_weights=[25, 2, 1])
    model_name='unet_3classes_weighted.hdf5'   
else:
    model = unet_model(n_classes=num_class, class_weights=[1.5 ,1] )
    model_name='unet_2classes_weighted.hdf5'
   

model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',verbose=1, save_best_only=True)

history = model.fit_generator(
	training_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=epochs,
	validation_data=validation_generator,
    validation_steps=50,
    callbacks=[model_checkpoint])

#Displaying curves of loss and accuracy during training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
	


results = model.predict_generator(validation_generator,val_samples//batch_size,verbose=1)

# results [results > 0.5] = 1
# results [results <= 0.5] = 0
results= np.argmax(results, axis=3)
results= np.expand_dims(results, axis=3)
saveResult("segmentation_results",results,flag_multi_class = True,num_class =num_class)




list_IDs=partition ['validation'] 
plt.close('all')




for i, ID in enumerate(list_IDs):
    im=np.array(Image.open('images_rgb/' + ID))
    img= np.expand_dims(im,axis=0)/255
    ypred=np.squeeze(model.predict(img))
    
    if num_class==2:
        ypred[ypred<0.5]=0
        ypred[ypred>=0.5]=1
        aux=np.array(Image.open('segmentation_gt_2/' + ID)) /255
        mask=aux[:,:,0:2]
        mask[:,:,1]=np.logical_not(aux[:,:,0])
		
    else:
        aux=np.array(Image.open('segmentation_gt_3/ring/' + ID)) /255
        mask=aux
        aux=np.array(Image.open('segmentation_gt_3/pupil/' + ID)) #max já é 1
        mask[:,:,0]=aux
        mask[:,:,2]=np.logical_not(np.logical_or(mask[:,:,0],mask[:,:,1]))

    
    mask= np.argmax(mask, axis=2)
    ypred = np.argmax(ypred, axis=2)
    
    result_image = segmentation.mark_boundaries(im, ypred, mode='inner')
    result_image = segmentation.mark_boundaries(result_image, mask, mode='inner', color=(1, 0, 0))
    
    plt.figure()
    plt.imshow(result_image)
    plt.title(ID + ' ' + str(i))
        
