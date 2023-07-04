Here we present the implementations of the YOLO, RetinaNet and EfficientDet networks.

## YOLO

Our code is adapted from https://github.com/experiencor/keras-yolo3

Download the pretrained weights for YOLO [spot-eyespot](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist14026_tecnico_ulisboa_pt/ERdKpxPsiPNNrKxsn9lj3lMBfwarD-eeI_K1p8QNEMHbpw?e=ahHMhz) model

From the [keras-yolo3-master folder](https://github.com/Margarida-Silveira/Butterfly_CNN/tree/main/Detection/keras-yolo3-master) run:

python predict.py -c config_spot_eyespot.json -i F:\yolo\Butterfly_images\test\images -o F:\yolo\Butterfly_images\test\output

## RetinaNet

Our code is adapted from https://github.com/fizyr/keras-retinanet

Download the pretrained weights for RetinaNet [spot-eyespot](https://drive.google.com/file/d/1sienQj3S3dXmkN4_sHiiaRa7tM4ipmzG/view?usp=sharing) model

From the [keras-retinanet](https://github.com/Margarida-Silveira/Butterfly_CNN/tree/main/Detection/keras-retinanet) directory run:

!pip install .

!python setup.py build_ext --inplace

[create_dataset.py](https://github.com/Margarida-Silveira/Butterfly_CNN/blob/main/Detection/keras-retinanet/create_dataset.py), (after changing ```DATASET_DIR```, which corresponds to the directory where the dataset is located containing the images and corresponding ground-truth annotations (.xml files))

!retinanet-evaluate --convert-model --config <Path to Config file> --gpu 0 csv ```Path to annotations.csv file``` ```Path to classes.csv file``` ```Path to the model .h5 file``` 

## EfficientDet

Our code is adapted from https://github.com/wangermeng2021/EfficientDet-tensorflow2

From the [efficientdet](https://github.com/Margarida-Silveira/Butterfly_CNN/tree/main/Detection/efficientdet) directory run:

!python3 detect.py --model-dir ```Path to the folder with the weights``` --tta False --pic-dir ```Path where the .jpg images are saved``` --save_dir ```Path to save the results``` --score-threshold 0.5 --class-names ```Path to the file class.names```
