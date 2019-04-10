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
The following documents a collection of models trained with this repository.
Links to the trained models and [output log files](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/logs/04_2019_logs.tar.gz) are provided.
The performance is evaluated on the validation set.

Note that all models here are not the original models used in paper,
but reproduced by this code base.
The reproduced performance reported here is very close to (or slightly better than) what's reported in paper,
but not exactly the same due to the stochastic nature of training.

#### AVA
| config | backbone | method | mAP | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ava_r50_baseline | R50-I3D-NL | 3D CNN | 22.2 | 102760666 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102760666/model_final.pkl) |
| ava_r50_lfb_avg | R50-I3D-NL | LFB-Avg | 23.3 | 103505104 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103505104/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103505104/lfb_model.pkl) |
| ava_r50_lfb_max | R50-I3D-NL | LFB-Max | 23.9 | 103505159 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103505159/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103505159/lfb_model.pkl) |
| ava_r50_lfb_nl | R50-I3D-NL | LFB-NL-2L | 25.8 | 102824705 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102824705/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102824705/lfb_model.pkl) |
| ava_r50_lfb_nl_3l | R50-I3D-NL | LFB-NL-3L | 25.9 | 106403526 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/106403526/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/106403526/lfb_model.pkl) |
| ava_r101_baseline | R101-I3D-NL | 3D CNN | 23.2 | 102760714 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102760714/model_final.pkl) |
| ava_r101_lfb_nl_3l | R101-I3D-NL | LFB-NL-3L | 26.9 (multi-crop: 27.7) | 105206523 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/105206523/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/105206523/lfb_model.pkl) |

#### EPIC Kitchens Verb
| config | backbone | method | top1 | top5 | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| epic_verb_r50_baseline | R50-I3D-NL | 3D CNN | 50.7 | 81.1 | 103704809 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103704809/model_final.pkl) |
| epic_verb_r50_lfb_avg | R50-I3D-NL | LFB-Avg | 52.9 | 82.5 | 103777391 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103777391/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103777391/lfb_model.pkl) |
| epic_verb_r50_lfb_max | R50-I3D-NL | LFB-Max | 53.3 | 81.0 | 103777432 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103777432/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103777432/lfb_model.pkl) |
| epic_verb_r50_lfb_nl | R50-I3D-NL | LFB-NL | 52.3 | 81.8 | 103777046 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103777046/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103777046/lfb_model.pkl) |

#### EPIC Kitchens Noun
| config | backbone | method | top1 | top5 | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| epic_noun_r50_baseline | R50-I3D-NL | 3D CNN | 26.2 | 51.0 | 104421642 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/104421642/model_final.pkl) |
| epic_noun_r50_lfb_avg | R50-I3D-NL | LFB-Avg | 29.1 | 56.3 | 103875866 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103875866/model_final.pkl) |
| epic_noun_r50_lfb_max | R50-I3D-NL | LFB-Max | 32.0 | 56.5 | 103875899 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103875899/model_final.pkl) |
| epic_noun_r50_lfb_nl | R50-I3D-NL | LFB-NL | 29.5 | 55.4 | 103706990 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103706990/model_final.pkl) |

#### EPIC Kitchens Action
| config | backbone | method | top1 | top5 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| epic_verb_r50_baseline & epic_noun_r50_baseline | R50-I3D-NL | 3D CNN | 19.4 | 38.1 |
| epic_verb_r50_lfb_avg & epic_noun_r50_lfb_avg | R50-I3D-NL | LFB-Avg | 21.2 | 41.3 |
| epic_verb_r50_lfb_max & epic_noun_r50_lfb_max | R50-I3D-NL | LFB-Max | 22.9 | 41.2 |
| epic_verb_r50_lfb_nl & epic_noun_r50_lfb_nl | R50-I3D-NL | LFB-NL | 21.8 | 40.5 |

**Note:** To make *action* predictions, we combine a *verb* model and a *noun* model, as opposed to training a separate *action* model.
Performance in this table is computed using the verb/noun models from the tables above.
Please see [GETTING_STARTED.md](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/GETTING_STARTED.md#evaluating-epic-kitchens-action-predictions)
for instructions on how to do this.


#### Charades
| config | backbone | method | mAP | model id | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| charades_r50_baseline | R50-I3D-NL | 3D CNN | 38.3 | 102766107 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102766107/model_final.pkl) |
| charades_r50_lfb_avg | R50-I3D-NL | LFB-Avg | 38.4 | 102999065 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102999065/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102999065/lfb_model.pkl) |
| charades_r50_lfb_max | R50-I3D-NL | LFB-Max | 38.6 | 102999121 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102999121/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/102999121/lfb_model.pkl) |
| charades_r50_lfb_nl | R50-I3D-NL | LFB-NL | 40.3 | 100866795 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/100866795/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/100866795/lfb_model.pkl) |
| charades_r101_baseline | R101-I3D-NL | 3D CNN | 40.4 | 103560426 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103560426/model_final.pkl) |
| charades_r101_lfb_avg | R101-I3D-NL | LFB-Avg | 40.8 | 103676713 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103676713/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103676713/lfb_model.pkl) |
| charades_r101_lfb_max | R101-I3D-NL | LFB-Max | 41.0 | 103676788 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103676788/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103676788/lfb_model.pkl) |
| charades_r101_lfb_nl | R101-I3D-NL | LFB-NL | 42.5 | 103641815 | [`model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103641815/model_final.pkl), [`lfb model`](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/103641815/lfb_model.pkl) |


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
