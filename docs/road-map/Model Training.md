# Model Training

Green boxes improve “accuracy”

Blue boxes improve inference speed, potentially at the cost of “accuracy”

## Overview 

Between each step, we should evaluate the model’s object detection performance, execution speed, and classification performance. If we find something acceptable, we can stop.

Efforts

Minimal - 1-2 days

Low - 3-5 days

Medium 1-2 weeks

Large - 2-8 weeks

## Public Weights

Effort: Minimal, asusming you have annotated data

Evaluate the following models. Code/weights here <https://github.com/google/automl/blob/master/efficientdet/README.md> 

Annotate your data with MS COCO labels <https://arxiv.org/pdf/1405.0312>

- EfficientDet d0 (512x512 input)
- EfficientDet d1 (512x512 input)
- EfficientDet d2 (640x640 input, see checkpoints below main list)
- EfficientDet d3 (640x640 input, see checkpoints below main list)
- EfficientDet Lite0 (320x320 input)
- EfficientDet Lite1 (320x320 input)
- EfficientDet Lite2 (320x320 input)
- EfficientDet Lite3 (320x320 input)
- EfficientDet Lite3 (320x320 input)
- EfficientDet Lite4 (320x320 input)

## Fine tune w/ Eco Data

Effort: Low

Hopefully, using above, we have a good idea of which model(s) fit our compute budget.  
For those models, using your data annotated in MS COCO format, start with the public weights, and fine-tune using only Eco data.

## Tune Anchor Boxes 

Effort: Minimal

<https://ecoparkingtechnologies.atlassian.net/wiki/spaces/EFV/pages/425361410/Object+Detection+Overview#Key-Term---Anchor-Box> 

[https://medium.com/@beam\_villa/enhance-object-detection-performance-through-anchor-box-optimization-761b68a1a4a4](https://medium.com/@beam_villa/enhance-object-detection-performance-through-anchor-box-optimization-761b68a1a4a4) 

For each class, generate the following distributions:

- Bounding Box Width
- Bounding Box Height
- Bounding Box Area
- Bounding Box Aspect Ratio

Don’t need to do K-Means as the article above, can just use intuition. Keep feature map receptive field in  mind when deciding which feature map should detect which object size. 

## Change Training Loop

Effort: Large

Change to PyTorch implementation at this point.

Repo:<https://github.com/rwightman/efficientdet-pytorch> 

Walkthrough: <https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f> 

Pytorch Lightning is to Pytorch, like Keras is to Tensorflow. Removes lots of boilerplate/behind the scenes work

<https://lightning.ai/docs/pytorch/stable/> 


Add Cyclic Learning Rate <https://paperswithcode.com/method/cyclical-learning-rate-policy> I prefer using triangular policies

Use Learning Rate test to find best learning rate

Use TorchVision and/or Kornia for augmentation

[https://pytorch.org/vision/main/auto\_examples/transforms/plot\_transforms\_getting\_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py)

[https://kornia.readthedocs.io/en/latest/applications/image\_augmentations.html](https://kornia.readthedocs.io/en/latest/applications/image_augmentations.html)  Can augment on GPU

implement Yolov4 Strategies <https://arxiv.org/pdf/2004.10934v1>

MixUp <https://paperswithcode.com/method/mixup> 

CutMix <https://paperswithcode.com/method/cutmix> 

CutOut <https://paperswithcode.com/method/cutout> I prefer using random data, rather than gray data

Use Focal Loss for class loss function <https://paperswithcode.com/method/focal-loss> 

Use Stochastic Depth <https://paperswithcode.com/method/stochastic-depth> 

Use CIoU for Bounding Box loss <https://learnopencv.com/iou-loss-functions-object-detection/> 


## Replace Backbone with MobileNet v4

Effort: Medium

Help with model inference run times

Paper: <https://arxiv.org/abs/2404.10518> 

Blog: <https://huggingface.co/blog/rwightman/mobilenetv4> 

Repo: <https://github.com/huggingface/pytorch-image-models> 

Repo License is Apache 2.0 - Free for commercial use, attribution required

[https://huggingface.co/docs/timm/feature\_extraction](https://huggingface.co/docs/timm/feature_extraction) 

## Prune Model

Effort: Medium / Large

Involves training multiple models.  
[https://proceedings.neurips.cc/paper\_files/paper/2021/file/bef4d169d8bddd17d68303877a3ea945-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/bef4d169d8bddd17d68303877a3ea945-Paper.pdf)

Paramaterize model construction:

- Number of feature maps to output 
    - Do we need to go down to P7/128 if we’re not detecting objects that large?
- Depth - number of layers in backbone
- Width - Number of Channels per block

Run some variants of model hyperparameters, either manually, or using hyperparameter optimization loop.  
[https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples\_hyperparam\_opt](https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples_hyperparam_opt) 

Use multi objective hyper parameter optimization

- Maximize mAP for vehicle class 
- Minimize FLOPS or inference time on RPi 
    - Recommend building basic python API hosted on raspberry pi that can accept model definition and execute on raspberry pi, returning benchmarks.


## Hyperparameter Optimization

[https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples\_hyperparam\_opt](https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples_hyperparam_opt) 

### Key Hyperparameters

Learning Rate - Use [https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/lr\_finder.html](https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/lr_finder.html) 

Model Regularization (L2)

Dropout Settings

Anchor Box Configuration  
RGB vs LAB

Preprocessing methods - CLAHE, etc
