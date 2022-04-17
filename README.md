# Tensorflow Object Detection Shell

## Prerequisites
- python
- pip

## Setup The Shell

### 1. Create a python virtual environment
``` 
python -m venv env
```

### 2. Activate the python virtual environment
```
# mac
source env/bin/activate
```
```
# windows
.\env\Scripts\activate
```

### 3. Install requirements
```
pip install -r requirements.txt
```

### 4. Run setup.py

```
python setup.py
```

### 5. Run the shell

```
python shell.py
```

## Train a model

### 1. Collect images for training
```
# Using all options
>>> collect --labels <label1> <label2> --limit <limit> --folder <images_folder>

# Using minimal options (limit=5, folder=collectedimages)
>>> collect --labels <label1> <label2>
```
### 2. Label collected images
```
# Using all options
>>> label --folder <images_folder>

# Using minimal options (folder=collectedimages)
>>> label
```

### 3. (Optional) Load a pretrained model
```
>>> load --url <pretrained_model_url>
```
- The shell already has [ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)
- You can get more pretrained models from [Tensorflow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

### 4. Train a model
```
# Using all options
>>> train --model <your_model_name> --labels <label1> <label2> --sample <sample_size> --pretrained <pretrained_model_name> --steps <number_of_steps> --folder <images_folder>

# Using minimal options (pretrained=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8, steps=2000m, folder=collectedimages )
>>> train --model <your_model_name> --labels <label1> <label2> --train <number_of_trainees>

```

### 5. Test trained model
```
# Test in realtime (via webcam)
>>> test --model <your_model_name>
```