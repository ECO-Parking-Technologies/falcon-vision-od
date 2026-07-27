# Data Preparation

Use CVAT as your source of truth for data models

Can use off the shelf model to assist with initial annotations

## Recommended Annotation Classes

Class recommendations based on publicly available dataset annotation classes. Aligning to these standards allows us to leverage their annotation guidelines, as well as making it easy to augment with public data if needed

### Vehicle - Something that will occupy a parking space

cars, trucks, buses, trailers, construction vehicles, motorcycle, bicycle, bicyclist

### Pedestrian - A person

### Animal (Optional) 

cat, dog, horse, cow, bear, deer, zebra  
NOT a bug in front of lens

## Annotation guidelines

Keep train/test/val split in CVAT, not randomly assigned in training loop

Stratify train/test/val based on garage and sensor camera, not raw image, see classifier generator

Keep images at full resolution as long as possible, downscale as a training pre-processing step

Prepare an annotation guideline document with examples for each situation

Keep annotations tight, no padding


All the tags / attributes below will help us evaluate our model

### Tag images <https://docs.cvat.ai/docs/manual/advanced/annotation-with-tags/> 

- garage name, 
- sensor name, 
- Time of day: 
    - morning - during sunrise
    - day - after sunrise, before sunset
    - evening: during sunset
    - night: between sunset and sunrise
- Special scenarios: Glare, snow, raining, bug in front of lens, dirty/fogged lens, etc

### Object Attributes <https://docs.cvat.ai/docs/manual/basics/attribute-annotation-mode-basics/>  

- Partial and/or Occluded - All classes, some external datasets might have these flags set too

Car on left is partial, car on right is occluded by pole. 

- InEcoParkingSpot - Only for your images, only for vehicle class
    - Mark this attribute if this car would be in one of the sensor monitored parking spots.
    - Ex: On a ramp, when you can see cars behind the main spots through the cables,  you would not mark this attribute for the cars on the ramp.
    - Ex2: you should annotate all cars in this image, but leave the InEcoParkingSpot attribute == false/null because the sensor would only be monitoring the three spots closest to the camera.
    - 
- InMotion - Only for your images, only for vehicle class
    - A car that is driving down the lane that is blocking the sensor spots

## Existing internal data sources

Old best practice was showing your model every image you had. New best practices emphasize showing more unique images vs the same image repeatedly. In theory, your augmentation pre-processing will add enough variation. Showing your model all images is still the safest option, but at the cost of longer training times, and potential over-fitting.

1. Static images for each sensor camera
    1. Show image if a car has left/entered spot
    2. Show at least 1 image per morning, day, evening, night
2. Export frames from motion security videos
    1. This is going to be a gold-mine
    2. Export and annotate all frames when objects are in motion.

## External data sources to augment

<https://paperswithcode.com/dataset/nuscenes>

<https://paperswithcode.com/dataset/kitti> 

<https://paperswithcode.com/dataset/waymo-open-dataset> 

<https://paperswithcode.com/dataset/coco> 

For each dataset, import images. You might need to transform the annotations from 3d or segmentation sets into bounding boxes, CVAT should be able to help  
You will also need to map your existing classes into your 3 classes
