# Model Selection

For inference at the edge, I recommend a model from the EfficientDet family.   
<https://arxiv.org/abs/1911.09070> 

Using any YOLO version past v4 requires a commercial license: <https://www.ultralytics.com/license> , otherwise you’re required to release open-source.  
  
I’m not a lawyer, but I would recommend ceasing use of YOLO v8/v11 immediately, even for internal tools / PoCs


EfficientDet-d0 requires input of 512x512

EfficientDet-d1-d3 requires input of 640x640  
EffiicentDet-Lite requires input of 320x320

Evaluate with off the shelf weights for efficientDet-d0, d1, d2, d3 and efficientDet-lite  
<https://github.com/google/automl/blob/master/efficientdet/README.md>   

If efficientDet doesn’t meet our compute requirements, recommend replacing backbone with MobileNet v4

<https://arxiv.org/abs/2404.10518>   
<https://huggingface.co/blog/rwightman/mobilenetv4> 

### Alternatives

FOMO - <https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices>
