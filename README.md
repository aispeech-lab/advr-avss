# audio-visual-speech-separation

## Abstract
Speech separation aims to separate individual voice from an audio mixture of multiple simultaneous talkers. Although audio-only approaches achieve satisfactory performance, they build on a strategy to handle the predefined conditions, such as the number of speakers is determined. Towards the cocktail party problem, we proposed a novel audio-visual speech separation model. In our model, we use the face detector to detect the number of speakers in the scene and use visual information to avoid the permutation problem. To improve our model's generalization ability to unknown speakers, we extract speech-related (speaker-independent) visual features from visual inputs explicitly by the adversarially disentangled method, and this feature is used to assist speech separation. Besides, the time-domain approach is adopted, which could avoid the phase reconstruction problem existing in the time-frequency domain models. To compare our model's performance with that of other models, we create two benchmark datasets of 2-speaker mixture from GRID and TCD-TIMIT audio-visual datasets. Through a series of experiments, our proposed model is shown to outperform the state-of-the-art audio-only model and three audio-visual models.

<div align=center><img width="400" src="https://github.com/ParrtZhang/AVSS_ADVR/blob/master/Image/Figure6.png" alt="The framework of our model"/></div>

## Motivation
We expect to elegantly resolve two problems in speech separation task in a unified audio-visual speech separation model: **permutation problem** and **unknown number of sources in the mixture**. Then, to **improve the model's generalization ability to unknown speakers**, we use the adversarially disentangled method to extract a **relative speaker-independent visual-speech feature** from face thumbnails. So our model can achieve excellent performance even on limited size datasets, which is a great advantage when data resources are limited.

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
