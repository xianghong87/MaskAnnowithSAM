# MaskAnnowithSAM
This project uses LabelImg to annotate detection boxes and then utilizes SAM to obtain corresponding segmentation contours. 

LabelImg is an open-source image annotation tool that facilitates object annotation in images and generates corresponding annotation files. In this project, LabelImg is used to annotate detection boxes in images. 

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.
