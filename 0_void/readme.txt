# Script for model development
## Training the model

cd yolov7-segmentation
python segment/train.py --data 0_void/custom.yaml --batch 1 --weights yolov7-seg.pt --cfg 0_void/yolov7-seg-1c.yaml --name void --img 640 --hyp hyp.scratch-high.yaml --device 0 --epochs 10

## Start Tensorboard session
### Please Open another terminal dir yolov7
$ cd yolov7-segmentation
$ tensorboard --logdir runs/train

## Inference with new model
$ cd yolov7-segmentation
$ python detect.py --weights runs/train/mark10/weights/best.pt --conf-thres 0.7 --source 0_mark_ocr/obj_train_data
