# Brain Tumor Segmentation
## Intro  
This repository contains implementation of the 3D-UResNet[1] model and was trained on MICCAI BRaTS dataset from Kaggle[2] that combines MRI images from 2020, 2019, and 2018 competitions. In contrast to UNet, UResNet, as the name suggests, employs residual blocks instead of concatenation to enable unhindered learning and use more layers.

## MRI Image and the segmented image  
### Original and Segmented Image
In the following image, the green color pixels represents the tumorous cells.  
<img alt="Segmented and the Original image" src="pred_gt.png" width="450" height="450" style="max-width: 100%;">

### Original Image  
<img alt="MRI Image and the segmented image" src="image_animation.gif" width="400" height="450" style="max-width: 100%;">

### Segmented Image  
<img alt="MRI Image and the segmented image" src="prediction_animation.gif" width="400" height="450" style="max-width: 100%;">

## References  
1. U-ResNet: Ultimate coupling of Registration and Segmentation with deep Nets https://link.springer.com/chapter/10.1007/978-3-030-32248-9_35
2. BRaTS: Brain Tumor Segmentation https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015
