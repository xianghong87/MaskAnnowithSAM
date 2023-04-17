# MaskAnnowithSAM
This project uses LabelImg to annotate detection boxes and then utilizes SAM to obtain corresponding segmentation contours. 

LabelImg is an open-source image annotation tool that facilitates object annotation in images and generates corresponding annotation files. In this project, LabelImg is used to annotate detection boxes in images. 

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

![alt text](https://github.com/xianghong87/MaskAnnowithSAM/blob/main/diagram.png?raw=true)

#### How to use
1. anno your images with LabelImg
2. change the path in options.py to your image and xml path.
3. try main.py to get your own anno mask

#### Installation
```
pip install matplotlib, numpy, opencv-python
pip install git+https://github.com/facebookresearch/segment-anything.git
```
