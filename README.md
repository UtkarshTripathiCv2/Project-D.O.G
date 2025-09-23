 Project-D.O.G

https://github.com/HumanSignal/labelImg/releases   FOR LABELIMG BUT IT GIVES IN XML SUITABLE FOR TENSORFLOW

 FOR LABELSTUDIO DOWNLOAD ANACONDA
 conda create --name yolo-env1 python=3.12
 conda activate yolo-env1
 pip install label-studio

 label-studio start
 THEN LABELSTUDIO FINALLY DOWNLOADED 


 google colab codes  https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 8
names: ['0', '1', '2', '3', '4', '5', '6', '7']

roboflow:
  workspace: test-ifmxt
  project: chili-plant-disease-1mhkt
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/test-ifmxt/chili-plant-disease-1mhkt/dataset/1  chilli
  fire and smoke 
  path: /kaggle/working/D Fire Dataset  # dataset root dir
train: data/train/images  # train images (relative to 'path')
val: data/val/images  # val images (relative to 'path')
test: data/test/images  # test images (relative to 'path')

# Classes
names: ['smoke', 'fire']  # Replace with your actual class names

# Counts
nc: 2  # number of classes
train_count: 14122
val_count: 3099
test_count: 4306
names:
- Apple Scab Leaf
- Apple leaf
- Apple rust leaf
- Bell_pepper leaf spot
- Bell_pepper leaf
- Blueberry leaf
- Cherry leaf
- Corn Gray leaf spot
- Corn leaf blight
- Corn rust leaf
- Peach leaf
- Potato leaf early blight
- Potato leaf late blight
- Potato leaf
- Raspberry leaf
- Soyabean leaf
- Soybean leaf
- Squash Powdery mildew leaf
- Strawberry leaf
- Tomato Early blight leaf
- Tomato Septoria leaf spot
- Tomato leaf bacterial spot
- Tomato leaf late blight
- Tomato leaf mosaic virus
- Tomato leaf yellow virus
- Tomato leaf
- Tomato mold leaf
- Tomato two spotted spider mites leaf
- grape leaf black rot
- grape leaf
nc: 30
roboflow:
  license: CC BY 4.0
  project: plantdoc
  url: https://universe.roboflow.com/joseph-nelson/plantdoc/dataset/1
  version: 1
  workspace: joseph-nelson
test: ../test/images
train: ../train/images
val: ../valid/images
  tomato
