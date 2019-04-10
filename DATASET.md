# Data Prepration

This document describes how we prepare AVA, EPIC-Kitchens, and Charades datasets.

**Note: After finishing the following steps, please verify that the images in `frames` are consistent with the "frame lists". Using frames at a different FPS or with a different resolution might result in different performance.**

### AVA
We assume that the AVA dataset is placed at `data/ava` with the following structure.
```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv
```
You can prepare this structure with the following steps
or by creating symlinks to your data.

1. Download videos
```Shell
cd dataset_tools/ava
./download_videos.sh
```
(These video files take 157 GB of space.)

2. Cut each video from its 15th to 30th minute
```Shell
./cut_videos.sh
```

3. Extract frames
```Shell
./extract_frames.sh
```
(These frames take 392 GB of space.)

4. Download annotations
```Shell
./download_annotations.sh
```

5. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv)) and put them in
the `frame_lists` folder (see structure above).

6. Download person boxes ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv), [test](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv)) and put them in the `annotations` folder (see structure above).
If you prefer to use your own person detector, please see details
in [GETTING_STARTED.md](GETTING_STARTED.md#ava-person-detector).


### EPIC Kitchens

We assume that the EPIC-Kitchens dataset is placed at `data/epic` with the following structure.
```
epic
|_ frames
|  |_ P01
|  |  |_ P01_01_000001.jpg
|  |  |_ ...
|  |_ ...
|  |_ P31
|     |_ P31_01_000001.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
|  |_ [official EPIC-Kitchens annotation files]
|_ noun_lfb
   |_ train_lfb.pkl
   |_ val_lfb.pkl
```
You can prepare this structure with the following steps
or by creating symlinks to your data.

1. Download videos with https://github.com/epic-kitchens/download-scripts/blob/master/download_videos.sh

2. Extract frames (please modify the script based on your data path. )
```Shell
cd dataset_tools/epic
./extract_epic_frames.sh
```
(These frames take 147 GB of space.)

3. Download annotations
```Shell
cd [path/to/video-lfb/root]
mkdir -P data/epic
git clone https://github.com/epic-kitchens/annotations.git data/epic/annotations
```

4. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/epic/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/epic/frame_lists/val.csv)) and put them in
the `frame_lists` folder (see structure above).

5. Download the pre-computed "Noun LFB" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/epic/noun_lfb/train_lfb.pkl), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/epic/noun_lfb/val_lfb.pkl)) and put them in the `noun_lfb` folder (see structure above).
If you prefer to train the detector yourself, please see details
in [GETTING_STARTED.md](GETTING_STARTED.md#epic-kitchens-noun-lfb).

### Charades

We assume that the Charades dataset is placed at `data/charades` with the following structure.
```
charades
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]-000001.jpg
|  |  |_ [video name 0]-000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]-000001.jpg
|     |_ [video name 1]-000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
```
You can prepare this structure with the following steps
or by creating symlinks to your data.

1. Download RGB frames from http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar
and put them in (or create symbolic link as) `data/charades/frames`.
(The downloaded tar ball takes 76 GB, and the extracted frames take 85GB of space. )

2. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/charades/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/charades/frame_lists/val.csv)) and put them in
the `frame_lists` folder (see structure above).
