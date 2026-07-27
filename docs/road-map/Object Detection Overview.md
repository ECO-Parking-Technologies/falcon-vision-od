# Object Detection Overview

## What is Object Detection

### Object Detection Neural Network Architectures

EfficientDet Architecture

1. The backbone extracts **Feature Maps** from the input image
2. The Feature Fusion layers combine info across different resolutions of filter maps
3. For each Anchor Box on each Feature Map
    1. The class head predicts the class of the object, car, truck, boat, person, etc. as a vector
    2. The box head predicts the actual bounding box size relative to the anchor box size
4. Post processing layers (not shown) remove any duplicates or overlapping boxes

### Key Term - Feature Map

**Feature Map**

In this image, the EfficientNet Layers labelled P1/2 - P7/128 are feature maps.   
They represent extracted information from the original image in a numerical form.  
One pixel on Layer P3/8 represents a 8x8 area on the original image.

One pixel on Layer P4/16 represents a 16x16 area on the original image, etc

Example feature map output for P1/2

Example feature map output from P5/32


### Key Term - Anchor Box

**Anchor Box**

A candidate region in the output feature maps.   
Each pixel on a feature map typically has somewhere between 3 and 10 anchor boxes.  
In the example below, The 8x8 feature map has 4 anchor boxes per pixel.

- scale 0.8, aspect ratio 1 = 6x6 area in original image
- scale 2,  aspect ratio 1 = 16x16 area in original image
- scale 1.5, aspect ratio 2 = 12x24 area in original image
- scale 1.5, aspect ratio 0.5 = 24x12 area in original image

Tuning anchor box sizing using your own data is a critical step

<https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html>
