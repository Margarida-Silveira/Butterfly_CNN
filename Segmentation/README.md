unzip the files in folders:
images_rgb
segmentation_gt_2
segmentation_gt_3

To train the U-net model:
python train.py 

To test the U-net model using pretrained models:
download the pretrained model weights for [2class](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist14026_tecnico_ulisboa_pt/EatZYXAzZ2NBqgvzl6Eb6lsBGLDaKsHL5Fm0GXpAXK5qtQ?e=AVhkfj) and [3class](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist14026_tecnico_ulisboa_pt/EXeR0-eHujZOsmwUzcRr1L0BWOkhoobfKbDx2y_XUkICEg?e=RWzuY4) models 
edit predict_unet.py and change num_class, input_path and save_path
python predict_unet.py
