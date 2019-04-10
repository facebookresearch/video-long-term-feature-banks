## Long-Term Feature Banks for Detailed Video Understanding ##

[Chao-Yuan Wu](https://www.cs.utexas.edu/~cywu/),
[Christoph Feichtenhofer](http://feichtenhofer.github.io/),
Haoqi Fan,
[Kaiming He](http://kaiminghe.com),
[Philipp Kr&auml;henb&uuml;hl](http://www.philkr.net/),
[Ross Girshick](http://rossgirshick.info)
<br/>
In CVPR 2019.
[[Paper](https://arxiv.org/abs/1812.05038)]
<br/>
<br/>

<div align="center">
<img src="figs/lfb_concept_figure.jpg" width="800">
</img></div>

<br/>
This is a Caffe2 based implementation
for our CVPR 2019 paper on Long-Term Feature Banks (LFB).
LFB provides supportive
information extracted over the entire span of a video, to
augment state-of-the-art video models that otherwise would
only view short clips of 2-5 seconds.
Our experiments
demonstrate that augmenting 3D CNNs
with an LFB yields state-of-the-art results
on AVA, EPIC-Kitchens, and Charades.

### Data Preparation and Installation
Please see [DATASET.md](DATASET.md), [INSTALL.md](INSTALL.md) for instructions.

### Training and Inference
Please see [GETTING_STARTED.md](GETTING_STARTED.md) for details.

### Results
The following documents models trained with this repository.
Links to the trained models as well as their output are provided.
We report performance evaluated on the validation set of each dataset.


#### AVA
| config | method | backbone | mAP | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ava_r50_baseline.yaml | 3D CNN | R50-I3D-NL | 22.2 | 102760666 | [`model`]() |
| ava_r50_lfb_avg.yaml | LFB-Avg | R50-I3D-NL | 23.3 | 103505104 | [`model`](), [`lfb model`]() |
| ava_r50_lfb_max.yaml | LFB-Max | R50-I3D-NL | 23.9 | 103505159 | [`model`](), [`lfb model`]() |
| ava_r50_lfb_nl.yaml | LFB-NL (default) | R50-I3D-NL | 25.8 | 102824705 | [`model`](), [`lfb model`]() |
| ava_r50_lfb_nl.yaml | LFB-NL-3L | R50-I3D-NL | 25.9 | 106403526 | [`model`](), [`lfb model`]() |
| ava_r101_baseline.yaml | 3D CNN | R101-I3D-NL | 23.2 | 102760714 | [`model`]() |
| ava_r101_lfb_nl_3l.yaml | LFB-NL-3L | R101-I3D-NL | 26.9 (multi-crop: 27.7) | 105206523 | [`model`](), [`lfb model`]() |

#### EPIC Kitchens Verb
| config | method | backbone | top1 | top5 | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| epic_verb_r50_baseline.yaml | 3D CNN | R50-I3D-NL | 50.7 | 81.1 | 103704809 | [`model`]() |
| epic_verb_r50_lfb_avg.yaml | LFB-Avg | R50-I3D-NL | 52.9 | 82.5 | 103777391 | [`model`](), [`lfb model`]() |
| epic_verb_r50_lfb_max.yaml | LFB-Max | R50-I3D-NL | 53.3 | 81.0 | 103777432 | [`model`](), [`lfb model`]() |
| epic_verb_r50_lfb_nl.yaml | LFB-NL | R50-I3D-NL | 52.3 | 81.8 | 103777046 | [`model`](), [`lfb model`]() |

#### EPIC Kitchens Noun
| config | method | backbone | top1 | top5 | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| epic_noun_r50_baseline.yaml | 3D CNN | R50-I3D-NL | 26.2 | 51.0 | 104421642 | [`model`]() |
| epic_noun_r50_lfb_avg.yaml | LFB-Avg | R50-I3D-NL | 29.1 | 56.3 | 103875866 | [`model`]() |
| epic_noun_r50_lfb_max.yaml | LFB-Max | R50-I3D-NL | 32.0 | 56.5 | 103875899 | [`model`]() |
| epic_noun_r50_lfb_nl.yaml | LFB-NL | R50-I3D-NL | 29.5 | 55.4 | 103706990 | [`model`]() |

#### EPIC Kitchens Action
| config | method | backbone | top1 | top5 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| epic_verb_r50_baseline.yaml & epic_noun_r50_baseline.yaml | 3D CNN | R50-I3D-NL | 19.4 | 38.1 |
| epic_verb_r50_lfb_avg.yaml & epic_noun_r50_lfb_avg.yaml | LFB-Avg | R50-I3D-NL | 21.2 | 41.3 |
| epic_verb_r50_lfb_max.yaml & epic_noun_r50_lfb_max.yaml | LFB-Max | R50-I3D-NL | 22.9 | 41.2 |
| epic_verb_r50_lfb_nl.yaml & epic_noun_r50_lfb_nl.yaml | LFB-NL | R50-I3D-NL | 21.8 | 40.5 |

#### Charades
| config | method | backbone | mAP | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| charades_r50_baseline.yaml | 3D CNN | R50-I3D-NL | 38.3 | 102766107 | [`model`]() |
| charades_r50_lfb_avg.yaml | LFB-Avg | R50-I3D-NL | 38.4 | 102999065 | [`model`](), [`lfb model`]() |
| charades_r50_lfb_max.yaml | LFB-Max | R50-I3D-NL | 38.6 | 102999121 | [`model`](), [`lfb model`]() |
| charades_r50_lfb_nl.yaml | LFB-NL | R50-I3D-NL | 40.3 | 100866795 | [`model`](), [`lfb model`]() |
| charades_r101_baseline.yaml | 3D CNN | R101-I3D-NL | 40.4 | 103560426 | [`model`]() |
| charades_r101_lfb_avg.yaml | LFB-Avg | R101-I3D-NL | 40.8 | 103676713 | [`model`](), [`lfb model`]() |
| charades_r101_lfb_max.yaml | LFB-Max | R101-I3D-NL | 41.0 | 103676788 | [`model`](), [`lfb model`]() |
| charades_r101_lfb_nl.yaml | LFB-NL | R101-I3D-NL | 42.5 | 103641815 | [`model`](), [`lfb model`]() |


### License
Video-long-term-feature-banks is Apache 2.0 licensed, as found in the LICENSE file.


### Citation
```
@inproceedings{lfb2019,
  Author    = {Chao-Yuan Wu and Christoph Feichtenhofer and Haoqi Fan
               and Kaiming He and Philipp Kr\"{a}henb\"{u}hl and
               Ross Girshick},
  Title     = {{Long-Term Feature Banks for Detailed Video Understanding}},
  Booktitle = {{CVPR}},
  Year      = {2019}}
```
