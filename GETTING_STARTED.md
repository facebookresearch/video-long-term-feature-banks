# Getting started

This document describes how to train and test models with our repository.
In the following we consider AVA with R50-I3D-NL backbone as an example.
Other datasets or backbones are trained/tested in a similar way.

Each experiment is defined by a configuration (.yaml) file.
The list of all yaml files are available in [README.md](README.md#results).

### Training

#### Download pre-trained weights
We pre-train all models on ImageNet-1K and Kinetics-400.
The pre-trained model weights are provided.
Please download ([R50-I3D-NL](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/r50_k400_pretrained.pkl), [R101-I3D-NL](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/r101_k400_pretrained.pkl)) and put/symlink them in `[video-lfb root]/pretrained_weights` folder.

#### Training a baseline model
Run `train_net.py` with a corresponding yaml file.
```Shell
python2 tools/train_net.py \
  --config_file configs/ava_r50_baseline.yaml \
  CHECKPOINT.DIR /tmp/baseline-output
```

#### Training an LFB model
It involves 2 steps:
1. Train a baseline model that will be used to infer LFB.
```Shell
python2 tools/train_net.py \
  --config_file configs/ava_r50_baseline.yaml \
  CHECKPOINT.DIR /tmp/lfb-nl-step1-output
```
2. Train an LFB model.
```Shell
python2 tools/train_net.py \
  --config_file configs/ava_r50_lfb_nl.yaml \
  LFB.MODEL_PARAMS_FILE [path to baseline model weight from step1] \
  LFB.WRITE_LFB True \
  CHECKPOINT.DIR /tmp/lfb-nl-step2-output
```
This command will first construct an LFB using the model weight `LFB.MODEL_PARAMS_FILE`, and then train the LFB-NL model.
Here we set `LFB.WRITE_LFB` as `True` to store the inferred LFB, so that we can use
`LFB.LOAD_LFB` and `LFB.LOAD_LFB_PATH` to reuse it next time.
LFB will be stored in training job output folder (`CHECKPOINT.DIR`).
Storing LFB takes about 3.3 GB of space for AVA, 1.1 GB for EPIC-Kitchens verbs, and 4.4 GB for Charades.

#### Linear scaling rule
All of our released configs make use of 8 GPUs.
If you will be using different number of GPUs for training,
please modify training schedules and hyperparameters according to the linear scaling rule. See [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) paper for details.


### Testing
To test/evaluate a model, use the config (yaml) file you used for training, and run `tools/test_net.py` with the model weight to be evaluated.

Testing a baseline model:
```Shell
python2 tools/test_net.py \
  --config_file configs/ava_r50_baseline.yaml \
  TEST.PARAMS_FILE [path to model weight]
```

Testing an LFB model:
```Shell
python2 tools/test_net.py \
  --config_file configs/ava_r50_lfb_nl.yaml \
  LFB.LOAD_LFB True \
  LFB.LOAD_LFB_PATH [path to folder that stores LFB] \
  TEST.PARAMS_FILE [path to model weight]
```
Here we assume we set `LFB.WRITE_LFB` as `True` during training,
so we can directly load the LFB without re-constructing it.
In this case, `[path to folder that stores LFB]` is the output folder of training job.

For AVA models, you can set `AVA.TEST_MULTI_CROP` as `True` to perform
multi-scale, 2-flip, and 3-crop testing.

### Dataset specific details

#### AVA Person Detector
To use our predicted person boxes for experiments, please download ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv), [test](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv)) and put them in `data/ava/annotations/`.

To train the person detector yourself, please use [Detectron](https://github.com/facebookresearch/Detectron)
with [e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x_ava.yaml](configs/detectron/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x_ava.yaml).
The model weight of our trained detector is available [[here]](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/67091280/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x_ava.yaml.13_26_25.49ooxNS5/output/train/ava_train/generalized_rcnn/model_final.pkl).
The model obtains 93.9 AP@50 on validation set.


#### EPIC-Kitchens Noun LFB

Our pre-computed "Noun LFB" is available for download ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/epic/noun_lfb/train_lfb.pkl), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/epic/noun_lfb/val_lfb.pkl)).
It contains the features of the top-10 objects extracted at 1 FPS from an object detector.
To train/test an LFB model on EPIC-Kitchens "Nouns",
please download and put (or symlink) them in `data/epic/noun_lfb`.

To train the detector yourself, please use
[Detectron](https://github.com/facebookresearch/Detectron)
with [epic_object_detector.yaml](configs/epic_object_detector.yaml).
This yaml file makes use of Visual Genome pre-trained model weights, which is available [[here]](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/vg_1600_pretrained.pkl)).
The final model weight is available [[here]](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/64398428/vg_1600_epic_detection.yaml.11_43_19.1MEYsROm/output/train/epic_train/generalized_rcnn/model_final.pkl).

#### Evaluating EPIC-Kitchens "action" predictions
After training a verb/noun model, a `epic_predictions_final.pkl` file will be automatically generated.
Please run
```Shell
python2 tools/evaluate_actions.py --verb_file [path to verb predictions] --noun_file [path to noun predictions]
```
to combine "verb" and "noun" predictions and evaluate their "action" accuracy.

#### Charades two-stage training
Note that we train LFB models for Charades in two stages:
We first train the backbone without LFB (i.e., like training a baseline model), and then add LFB and continue training for half of the schedule.

Stage 1:
```Shell
python2 tools/train_net.py \
  --config_file configs/charades_r50_baseline.yaml \
  CHECKPOINT.DIR /tmp/lfb-nl-stage1-output
```

Stage 2:
```Shell
python2 tools/train_net.py \
  --config_file configs/charades_r50_lfb_nl.yaml \
  TRAIN.PARAMS_FILE [path to model weight from Stage 1] \
  LFB.MODEL_PARAMS_FILE [path to model weight for LFB inference] \
  LFB.WRITE_LFB True \
  CHECKPOINT.DIR /tmp/lfb-nl-stage2-output
```
