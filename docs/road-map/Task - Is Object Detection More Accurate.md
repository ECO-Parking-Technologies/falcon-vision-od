# Task \- Is Object Detection More Accurate?

You have a two layered evaluation to make. An object detection model answers “Where are cars?”, but your actual question is “Which parking spots are occupied?”  
  
We need to know if we can beat current classifier’s “accuracy” with object detection.  
  
See <https://ecoparkingtechnologies.atlassian.net/wiki/spaces/EFV/pages/425459716/Model+Prediction+Evaluation?atlOrigin=eyJpIjoiM2Q3Y2MzNjIwMjdhNDk2ZGFmY2RjYzRlMTM1MjQwZTkiLCJwIjoiYyJ9>  for tips on evaluating classifier performance

Build a python script that can do the following. We’ll need it in c++ eventually, so use OpenCV as much as possible for easy translation.

  
For object detection algo, use best off the shelf option (DETR or EfficientDet, see note about not using YOLO due to licensing in model selection)

Inputs

- Object detection output
- Spot definition file 

Output

For each spot, occupancy score between 0 and 1

Build an outer loop script that goes through as many spots as possible. Use a ClearML task, so we can directly compare the data to old classifier models.

  
Suggestions for algorithm:   
<https://machinelearningspace.com/intersection-over-union-iou-a-comprehensive-guide/> 

any IoU \> 0.5 == occupied space. Might need to tweak this threshold.   
For output, scale IoU to 0 - T =\> 0 - 1 and clamp to 0 - 1, where T = threshold of 0.5  

Red = old spot definition

Blue = object detection output

Green = suggested new spot definition (either polyline or rotated bounding box)  

If IoU with axis-aligned bounding boxes doesn’t suffice, I would recommend changing spot annotations to the actual footprint of the parking spot (green) and just using how much of spot definitino is covered by car.  Maybe weight pixel values near center of polyline near 

Other options to explore

- Oriented object detection / rotated object detection  
- <https://paperswithcode.com/task/3d-object-detection> 3D bounding boxes and 3D object detection, Find a model that works on just camera data, not a point cloud.  

  


To generate comparison metrics, use this function: [https://github.com/ECO-Parking-Technologies/falcon-vision-ml/blob/master/training\_pipeline/dnn/dnn\_validation.py#L13](https://github.com/ECO-Parking-Technologies/falcon-vision-ml/blob/master/training_pipeline/dnn/dnn_validation.py#L13)
