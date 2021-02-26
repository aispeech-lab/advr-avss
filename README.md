# Audio-Visual Speech Separation with Visual Features Enhanced by Adversarial Training

## Overview
Demo samples of our paper *Audio-Visual Speech Separation with Visual Features Enhanced by Adversarial Training*. 

We will release the whole code soon. If you have any question about implementation details, feel free to ask me (zhangpeng2018@ia.ac.cn).

## Abstract
Audio-visual speech separation (AVSS) refers to separating individual voice from an audio mixture of multiple simultaneous talkers by conditioning on visual features. **For the
AVSS task, visual features play an important role, based on which we manage to extract more effective visual features to improve the performance**. In this paper, we propose a novel AVSS model that uses speech-related visual features for isolating the target speaker. Specifically, the method of extracting speech-related visual features has two steps. Firstly, we extract the visual features that contain speech-related information by learning joint audio-visual representation. Secondly, we use the adversarial training method
to enhance speech-related information in visual features further. We adopt the time-domain approach and build audio-visual speech separation networks with temporal convolutional neural network block. Experiments on audio-visual datasets, including GRID, TCD-TIMIT, AVSpeech, and LRS2, show that our model significantly outperforms previous state-of-the-art AVSS models. We also demonstrate that our model can achieve excellent speech separation performance in noisy real-world scenarios. **Moreover, in order to alleviate the performance degradation of AVSS models caused by the missing of some video frames, we propose a training strategy, which makes our model robust when video frames are partially missing**. 

%<div align=center><img width="400" src="https://github.com/aispeech-lab/advr-avss/blob/master/image/Figure2.png" alt="The framework of our model"/></div>

## Model
### Extract visual-speech feature by adversarial disentangled method
The model can be found at [*./visual_model*](./visual_model)
<div align=center><img width="500" src="./Image/Figure4.png" alt="Visual model of extracting visual-speech feature"/></div>

### Audio-visual speech separation networks
The model can be found at [*./ss_model*](./ss_model)
<div align=center><img width="500" src="./Image/Figure5.png" alt="Audio-visual speech separation networks"/></div>

## Datasets
[*./Datasets*](./Datasets) includes two benchmark datasets (train list, valid list, test list) of 2-speaker mixture from GRID and TCD-TIMIT audio-visual datasets. There is no overlapping speakers between the sets. To construct a 2-speaker mixture, we randomly choose two different speakers first, randomly select audio from each chosen speaker, and finally mix two audios at a random SNR between -5 dB and 5 dB. train set: 30 hours, valid set: 2.5 hours, test set: 2.5 hours.

## Result
### Video and Audio Samples
- Listen and watch the samples at [*./Samples*](./Samples).

## Pipeline
The pipeline of this project will be released soon. If you have any question about implementation details, feel free to ask me (zhangpeng2018@ia.ac.cn)
