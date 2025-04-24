# Falcon Vision Object Detection (EfficientDet)

This repository is a fork of `rwightman/efficientdet-pytorch`, adapted for use in the **Falcon Vision** parking guidance system. The goal is to transition from classification-based vehicle detection to **object detection** using **EfficientDet** for improved accuracy, scalability, and real-time performance on embedded devices.

## Purpose

- Detect vehicles and pedestrians in real-time on embedded hardware.
- Determine parking space occupancy using IoU with predefined zones.
- Train and fine-tune custom EfficientDet models using CVAT-annotated datasets.

## Features

- Support for training on custom datasets in **MS COCO** format.
- Compatible with **PyTorch Lightning** for simplified experimentation.
- Includes pre-trained EfficientDet models (Lite0–Lite4, D0–D7).
- Easily extendable to use different backbones (e.g., EfficientNetV2, MobileNetV4).
- Optimized for edge deployment (convert to ONNX / TFLite).

## FalconVision Setup Steps

1. **Annotation**: Use CVAT with `Vehicle`, `Pedestrian`, and custom attributes (e.g., `InEcoParkingSpot`, `InMotion`).
2. **Pre-annotation**: Use EfficientDet-Lite0 to pre-label raw images.
3. **Training**:
   - Fine-tune public EfficientDet weights on FalconVision images.
   - Optimize model for speed and accuracy on CM3+ hardware.
4. **Evaluation**:
   - Use ClearML + FiftyOne for metric tracking and visual diagnostics.

## Based On

- [Official EfficientDet (TensorFlow)](https://github.com/google/automl)
- [EfficientDet: Scalable and Efficient Object Detection (Paper)](https://arxiv.org/abs/1911.09070)
- [Training EfficientDet with PyTorch Lightning (Microsoft)](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f)

## Original README

The original documentation from `rwightman/efficientdet-pytorch` can be found [here](https://github.com/rwightman/efficientdet-pytorch/blob/master/README.md).