# Document Verification using multi-agent AI system

## Overview

This project is a proof of concept to show that we can create a multi-agent AI system for verfication and digitization of various legal documents and forms. We specifically focus on bank cheques and verifying whether they are authentic or not, as well as whats the account number of cheque issuer.

## Table of Contents
+ Overview
+ Table of Contents
+ Datasets
+ Preprocessing
+ Training
+ Evaluation
+ Usage
+ Dependencies

## Datasets

The following datasets have been using in training and evaluation of the model.

### CEDAR

CEDAR signature database contains signatures of 55 signers belonging to various cultural and professional backgrounds. It has been used to train the signature verification model. Each of these signers signed 24 genuine signatures 20 minutes apart. Each of the forgers tried to emulate the signatures of 3 persons, 8 times each, to produce 24 forged signatures for each of the genuine signers. Hence the dataset comprise 55 × 24 = 1320 genuine signatures as well as 1320 forged signatures.
<br><br>
Link for [*Dataset*](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)

### IDRBT 

IDRBT Cheque dataset was created by RBI and contains images of 112 filled in cheques belonging to 4 different indian banks. The dataset was used to train the extraction model which seperates out various useful components within the cheque.
<br><br>
Link for [*Dataset*](https://www.idrbt.ac.in/idrbt-cheque-image-dataset/)

## Preprocessing

### Verification

* The custom class for dataset to be used for this project has been given `utils.py` by the name *SiameseDataset*.
* The transform function is added into the class as a function that converts image to torch tensors and resizes them.
* If its part of train set then for data augmentation purposes randomized horizontal and vertical flips are also added.

### Extraction

* For extraction all preprocessing was done by YOLO class of ultralytics library itself.

## Training

### Verification

* The model has been trained on a trimmed down dataset of 20,884 entries with a validation set of 2984 entries.
* The architecture of models is given in `src/models/cnn.py`
* The code for Model trainer class is given in `src/models/train.py`
* Model has been trained using *AdamW optimizer* for 30 epochs with an initial learning rate of 3e-4 and early if loss does not improve for 6 epochs.
* For the first 15 epochs the learning rate reduces 15% per epoch.
* Custom Contrastive Loss function has been used which is availabe in `src/utils/loss.py`

### Extraction

* Model was trained using command-line tool of YOLO for 100 epochs with early stopping if loss does not improve for 15 epochs.

## Evaluation

### Verification

The model has been tested on testing data by setting a euclidean distance threshold value of 0.15 for smaller version to model mentioned in original paper, 0.242 for shufflenet, below it the signatures are authentic, above it there is a forgery. The code for this task is given in `src/models/evaluate.py`

```
For shufflenet
{
    "0": {
        "precision": 0.9259771705292287,
        "recall": 0.945268361581921,
        "f1-score": 0.9355233269264371,
        "support": 2832.0
    },
    "1": {
        "precision": 0.942099364960777,
        "recall": 0.9217836257309941,
        "f1-score": 0.9318307777572511,
        "support": 2736.0
    },
    "accuracy": 0.9337284482758621,
    "macro avg": {
        "precision": 0.9340382677450028,
        "recall": 0.9335259936564575,
        "f1-score": 0.9336770523418441,
        "support": 5568.0
    },
    "weighted avg": {
        "precision": 0.9338992833102481,
        "recall": 0.9337284482758621,
        "f1-score": 0.9337088846622681,
        "support": 5568.0
    }
}
```

```
For custom model
{
    "0": {
        "precision": 0.9811937857726901,
        "recall": 0.847457627118644,
        "f1-score": 0.9094353921940129,
        "support": 2832.0
    },
    "1": {
        "precision": 0.8616271620755925,
        "recall": 0.9831871345029239,
        "f1-score": 0.9184021850460908,
        "support": 2736.0
    },
    "accuracy": 0.9141522988505747,
    "macro avg": {
        "precision": 0.9214104739241413,
        "recall": 0.915322380810784,
        "f1-score": 0.9139187886200519,
        "support": 5568.0
    },
    "weighted avg": {
        "precision": 0.9224412206801508,
        "recall": 0.9141522988505747,
        "f1-score": 0.9138414886816719,
        "support": 5568.0
    }
}
```

![reports\figures\shufflenet_cf.png](Models\Verfication\reports\figures\shufflenet_cf.png)
![reports\figures\custom_cf.png](Models\Verfication\reports\figures\custom_cf.png)

### Extraction

We have finetuned YOLO on the IDRBT dataset and analysed its performance on the usual object detection metrics such mAP.


![Models\Extraction\reports\detect\train7\confusion_matrix_normalized](Models\Extraction\reports\detect\train7\confusion_matrix_normalized.png)
![Models\Extraction\reports\detect\train7\PR_curve](Models\Extraction\reports\detect\train7\PR_curve.png)
![Models\Extraction\reports\detect\train7\results](Models\Extraction\reports\detect\train7\results.png)

```
Precision: 0.96
Recall: 0.97
mAp(50): 0.96
mAP(95): 0.73
```

## Usage 

To run the project simply enter into terminal: `python app.py` and hit enter.

## Dependencies

All dependencies can be installed by running the following command in the terminal:

```pip install -r requirements.txt```