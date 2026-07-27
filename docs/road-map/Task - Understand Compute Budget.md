# Task \- Understand Compute Budget

For ML at the edge, we have to balance model “accuracy” against compute time.  
We can have a perfect object detection model, but if it uses all 4 threads and takes 25 seconds, nothing else will run on the sensor.   
  
Our objective here is just to get rough order of magnitude estimates on runtimes.  
  
Purchase one or more Raspberry Pi 3s (3B?) that correspond to your compute module.  
Recommend using python to evaluate vs c++ at this stage. The python code is calling c++ version under the hood.   
  
  
Try benchmarking tensorflow lite vs pytorch torchscript versions too. See this issue on c++ tensorflow being slower than python <https://github.com/tensorflow/tensorflow/issues/55476>   
  
**1. Benchmark efficientdet-d0, d1, d2, d3 and all efficientdet-lite models on device, run 1000 times. Can use random data, no need to actually feed images to model.**

Make sure to set thread count = 1 for best comparison

Tensorflow lite checkpoints here: <https://github.com/google/automl/blob/master/efficientdet/README.md> 

Note, use the versions of D2-D3 that are trained at 640x640 <https://github.com/google/automl/blob/master/efficientdet/README.md?plain=1#L101> 

pytorch implementation here: <https://github.com/rwightman/efficientdet-pytorch> 

Raspberry Pi Torchscript walkthrough: [https://pytorch.org/tutorials/intermediate/realtime\_rpi.html](https://pytorch.org/tutorials/intermediate/realtime_rpi.html) 




Task 2 - Evaluate using Raspberry Pi GPU for model inference tasks:

Would allow for much more powerful models to run on your existing hardware base with no additional hardware (ex: coral compute stick)  
I know this is experimental, but is seriously worth considering if we can get it to work.

<https://docs.broadcom.com/doc/12358545> - GPU documentation

 <https://github.com/doe300/VC4CL>   
<https://github.com/Idein/py-videocore>
