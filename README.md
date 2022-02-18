# On-Cloud-N: Cloud Detection Challange

!!! WORK IN PROGRESS !!!

## Hardware requirements
- Run on Google Colab Pro
- GPU (model or N/A): 1x Tesla P100 with 16GB 
- Memory (GB): 1x 168GB
 OS: Linux
- CPU RAM: 1x 16 GB
- CUDA Version : 11.2
- Driver Version: 460.32.03
- Disk: 128 GB

## Softwae requirements
Required software are listed on requirements.txt. Please install all the dependencies before executing the pipeline.

## How to run
You can check the run.ipynb notebook for the main point of entry to my code.

### Data preparation
First, the training and testing data should be downloaded from the competition website. Ideally, the data can be placed in the data folder in the repo directory. The repo tree would then look like below:

```
../on-cloud-n/
├── LICENSE.md
├── README.md
├── configs
│   ├── checkpoints.json
│   └── deep-chimpact.yaml
├── data
│   ├── train_features
│   │   ├── train_chip_id_1
│   │   │   ├── B02.tif
│   │   │   ├── B03.tif
│   │   │   ├── B04.tif
│   │   │   └── B08.tif
│   │   └── ...
│   └── train_labels
│       ├── train_chip_id_1.tif
│       ├── ...
│       ...
├── train_metadata.csv
...
```

**prepare_data.py**
- **--data-dir** directory for raw data (unprocessed images), default 'data/raw'
- **--debug** uses only 100 images for processing if this mode id used
- **--infer-only** generates images only for test 


### Training
Run train.py to train final 3 models using appropriate arguments.
**train.py**
- **--cfg** config file path
- **--fast-dev-run** trains only with 1 batch of the entire files
- **--model-name** name of the model
- **--img-size** image size. e.g. --img-size 512 512
- **--batch-size** batch size
- **--selected-folds** selected folds for training. e.g. --selected-folds 4
- **--all-data** use all data for training. No validation data
- **--ds-path** dataset path
- **--output-dir** path to save model weights and necessary files

## Training Pipeline
```
!python train.py --fast-dev-run 1 --cfg './configs/resnet34-unet-512.yaml'
!python train.py --fast-dev-run 1 --cfg './configs/efficientnet-b1-unet-512.yaml'
!python train.py --fast-dev-run 1 --cfg './configs/resnext50_32x4d-unet-512.yaml'
```

### Prediciton
Run predict.py in order to predict on test images.

#### predict.py
- **--model-dir** the directory where the models listed in are located. The * complete model location is model-dir/*.pt.
- **--ensemble** ensemble mode (0 = off, 1 = on)
- **--fast-dev-run** predicts only with 1 batch of the entire files
- **--output-dir** output folder to to save submission file
- **--tta** number of TTA's
- **--batch-size** batch size

## Infer Pipeline
* **Infer without Training:** First download the checkpoints from [here](https://www.kaggle.com/maxschfer/ocn-checkpoints) and place them on `./output` directory then run the following codes.
```
!python predict.py
```

* **Infer after Training:** After training is done, run the following codes.
```
!python predict.py
```

> Before predict, file tree would look like this:
```
../on-cloud-n/
...
├── data
│   ├── train_features
│   ├── train_labels
│   ├── test_features
│   └── train_metadata.csv
...
├── output
│    ├── Resnet34-Unet-512x512
│    ├── EfficientNetB1-Unet-512x512
│    └── Resnext50-Unet-512x512
...
 
```