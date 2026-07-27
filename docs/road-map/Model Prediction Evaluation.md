# Model Prediction Evaluation

You have a two layered evaluation to make. An object detection model answers “Where are cars?”, but your actual question is “Which parking spots are occupied?” 

<https://ecoparkingtechnologies.atlassian.net/wiki/spaces/EFV/pages/425623607/Is+Object+Detection+More+Accurate?atlOrigin=eyJpIjoiMWY5MDE2YjhiNDMzNGRmM2E4MjU2OTBkNzIwMjI3ZTUiLCJwIjoiYyJ9> 

I recommend creating two python scripts. This allows you to tweak function that does binary clasification without training/evaluating a new object detection model.

1. Generate metrics for object detection 
2. Given object detection model and a spot definition, compute binary classification metrics for each spot.

# Object Detection Evaluation

[https://docs.voxel51.com/tutorials/evaluate\_detections.html](https://docs.voxel51.com/tutorials/evaluate_detections.html) 

[https://docs.voxel51.com/user\_guide/app.html#app-model-evaluation-panel](https://docs.voxel51.com/user_guide/app.html#app-model-evaluation-panel) 

For each object class, and in the aggregate

Confusion Matrix

Precision Recall curve

Area under Precision Recall curve

Mean Average Precision  

## Binary Classifier Evaluation

Use existing metrics, see old classifier code for examples  
[https://github.com/ECO-Parking-Technologies/falcon-vision-ml/blob/master/training\_pipeline/dnn/dnn\_validation.py#L13](https://github.com/ECO-Parking-Technologies/falcon-vision-ml/blob/master/training_pipeline/dnn/dnn_validation.py#L13)
