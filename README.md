[![Unit Tests](https://github.com/mhuen/HDRHybridButterflies/actions/workflows/test_suite.yml/badge.svg)](https://github.com/mhuen/HDRHybridButterflies/actions/workflows/test_suite.yml)
[![codecov](https://codecov.io/gh/mhuen/HDRHybridButterflies/graph/badge.svg?token=WUUXF6JCHG)](https://codecov.io/gh/mhuen/HDRHybridButterflies)
[![DOI](https://zenodo.org/badge/910106609.svg)](https://doi.org/10.5281/zenodo.19241508)


# HDRHybridButterflies: Feature-based solution for Hybrid Butterfly Detection

This repository provides a solution for the [2024 HDR Anomaly Challenge: Hybrid Butterfly Detection](https://www.codabench.org/competitions/3764/). The solution uses a feature-based approach to detect anomalies. Individual classifiers are trained to detect certain features on the wings such as specific shapes and/or colors. A final anomaly score is then produced by evaluating the likelihood that the combination of found features on the upper and lower wings belongs to one of the subspecies in the training data. If it does not, then it is likely a hybrid butterfly.

The idea behind this approach is that the hybrid offspring will retain certain visible features from their parent subspecies, often combining them in a new way. If a robust model is found to detect and classify the presence of each of the features present in the parent subspecies, then this should lead to a robust detection of hybrid butterflies as long as they show a novel combination of these features.

The benefit of this approach is that the produced anomaly score is easily understood and interpretable. The individual scores for each feature directly indicate why a certain decision was made by the model and based on what visual properties of the wings.


# Note on ToDos

The models provided in this repository only use a crude and preliminary description of features of the parent subspecies. The trained classifiers are able to detect these defined features, but it is worth emphasizing that the labeling of these features for the training data is likely incorrect, leading to degraded results in the challenge. Nevertheless, given that this approach already yields promising results, it makes sense to define the visible features of the parent subspecies more rigorously, in particular with added domain knowledge from experts in the field.

# Repository Structure

```
|
|── data:
|       Contains the training data csv files from the challenge
|
|── hdr_hybrid_butterflies:
|       This contains the content of the pip installable python package
|
|── resources
|   |── notbookes:
|   |       Contains a notebook that was used to explore the data and
|   |       also to create more specific training data for the
|   |       individual feature classifiers.
|   |── scripts
|   |   |── create_submission.sh:
|   |   |        This script is used to generate the submission.zip file
|   |   |── model.py:
|   |   |        Defines the model class used in the submission.
|   |   |── ...
|   |   |        Additional files relevant to create the submission file.
|
|── tests:
|       Directory for unit tests
|
|── submission_final.zip:
|       The final submission file that was used in the challenge.
```

# Internal models

The repository utilizes a variety of models trained for different purposes. Initial models are responsible for extracting the relevant image segments containing only the butterfly wings. Subsequent models are trained and applied based on these extracted segments. This is done to reduce noise by only focusing on the relevant parts of the image.

- segmentation model: this model segments images to identify the individual wings while also classifying into upper and lower wings

- segment classifier: this model classifies an image segment into one of three categories: noise, upper wing, lower wing

- hybrid classifier: a model trained on labeled training data to differentiate the provided parent subspecies and signal hybrid class

- upper wing feature classifiers: a number of classifiers for each defined visible feature in the upper wing

- lower wing feature classifiers: a number of classifiers for each defined visible feature in lower wing

# References


[References for the training data of the challenge](butterfly_anomaly.bib)

This repository also used the [Grounding DINO + SAM combination by Niels Rogge](https://github.com/NielsRogge/Transformers-Tutorials/
blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb) to generate training
data for the internal wing segmentation model. This uses the [Grounding DINO]{https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino} and [SAM]{https://huggingface.co/docs/transformers/en/model_doc/sam} models under the hood, based on the [Grounded Segment Anything project]{https://github.com/IDEA-Research/Grounded-Segment-Anything}
