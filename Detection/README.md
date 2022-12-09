Here we present the implementations of the YOLO, RetinaNet and EfficientDet networks.

## YOLO

Our code is adapted from https://github.com/experiencor/keras-yolo3

Download the pretrained weights for YOLO [spot-eyespot](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist14026_tecnico_ulisboa_pt/EXeR0-eHujZOsmwUzcRr1L0BWOkhoobfKbDx2y_XUkICEg?e=RWzuY4) model

From the ```keras-yolo3-master folder``` run:

python predict.py -c config_spot_eyespot.json -i F:\yolo\Butterfly_images\test\images -o F:\yolo\Butterfly_images\test\output

## RetinaNet

Our code is adapted from https://github.com/fizyr/keras-retinanet

Download the pretrained weights for RetinaNet [spot-eyespot](https://drive.google.com/file/d/1GrliyIifPXJRyeWgGgoIzKVAhkJYxI8p/view?usp=sharing) model

From the ```retinanet/keras-retinanet``` directory run:

!retinanet-evaluate --convert-model --config <Path to Config file> --gpu 0 csv <Path to annotations.csv file> <Path to classes.csv file> <Path to the model .h5 file> 

## EfficientDet

Our code is adapted from https://github.com/wangermeng2021/EfficientDet-tensorflow2

