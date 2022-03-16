unzip the files in folders:
images_rgb
segmentation_gt_2
segmentation_gt_3

To train the U-net model:
python train.py 

To test the U-net model:
edit predict_unet.py and change num_class, input_path and save_path
python predict_unet.py
