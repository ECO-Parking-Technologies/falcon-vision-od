# Object Detection Roadmap

# Objectives

Replace existing parking occupancy neural net with object detection model that classifies all spots simultaneously.  
 

# Immediate Questions  

**Will Object Detection have better “accuracy” than current model?**

Note, for convenience sake, I will refer to overall model predictive capability as “accuracy”, but this is an all-encompassing term that includes metrics such as accuracy, TRP, TNR, Precision, Recall, etc.

An object detection model answers “Where are cars in this picture?”, but your actual question is “Which parking spots are occupied?” If we can’t get an object detection model that classifies the parking spots better than the existing model, should we even proceed?  

**What is our compute budget?**

What is our upper limit on execution time for an on-sensor object detection model?

How frequently should we run the model per camera?  

# Steps to Train/Fine-tune object detection model

Machine Learning training is an iterative process. Within each segment, I’ll provide an initial recommendation, with additional steps we can take if the model is not performing as desired

# Recommended Tools  

## CVAT

You’re already using this. I recommend maintaining your Train/Test/Val splits within CVAT.

**MIT License - Free for commercial use**  

## FiftyOne

<https://github.com/voxel51/fiftyone>

Visualize datasets and predictions, mark images for re-annotation, integrates with CVAT.

Has built-in evaluation metrics for object detection models

**Apache 2.0 license - Free for commercial use **  

## ClearML

Maintain model artifacts, manage experiments, queue training runs

I recommend upgrading to latest version and leveraging some of the orchestration features, like queue multiple model training runs without manually starting to keep the GPU running 100% of the time.  
[https://clear.ml/docs/latest/docs/clearml\_agent/](https://clear.ml/docs/latest/docs/clearml_agent/)  - Work execution agent

**On-prem open-source version free for commercial use**  

## PyTorch

<https://pytorch.org/> 

Neural network training library

Has edge inference capabilities (TorchScript)

Can convert models to ONNX, and ultimately to Tensorflow Lite models via ONNX

**Modified MIT License - Free for commercial use**
