# Learning-of-Image-Dehazing-Models-for-Segmentation-Tasks

PyTorch code for the paper **Learning of Image Dehazing Models for Segmentation Tasks** (https://arxiv.org/pdf/1903.01530.pdf)<br/> 

## **Approach:**<br/>
The generator network receives an image with haze as an
input and gives a candidate of a dehazed image as the output. Similar to the single image dehazing model, the generator loss
is computed through LGAN , Lpixel, Lpercep and Lseg. . LGAN is the loss function from Isola et al. used to generate fake images. Lpixel is the
reconstruction loss between the ground truth for dehazing (a.k.a. the real image) and the fake dehazed image, based on their
individual pixel values, allowing the network to produce crisper images. Lpercep is the perceptual loss used for preserving
important semantic elements of the image in the output of the generator. The segmentation loss Lseg, is computed by placing the output of
the generator (i.e., the dehazed image) into the segmentation network. The obtained segmentation map is then compared to
the ground truth segmentation map, using the L2 loss. Basically, the model tries at the same time to remove haze as much as
possible while preserving, or even improving segmentation performance.

## **What this paper propose ?**<br/>
This paper demonstrates the usefulness of including segmentation loss in an end-to-end training of deep learning
models for dehazing. The learning-based dehazing model is generated not just for denoising metrics, but also with an
optimization criterion aimed at achieving something useful for a specific task, and with performance improvements that can
be significant in comparison to results obtained with an unguided approach. Moreover we can consider to boost even more
the performance of DFS using directly an approximation of the IoU/iIoU measures for gradient descent, which are better
optimization measure than mean square error and similar.

## **How to run train_DFS.py ?**<br/>
1) Follow the procedure to make the Foggy Cityscape dataset (https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/).<br/>
2) Make a folder called cityscape, the path to that folder is your "path_exp", make 3 sub-folders: a,b and c. Put the hazy images in a, the non-hazy image in b and the segmentation mask in c.<br/>
3) Change the "path_exp" in train_DFS.py to your real experimentation path. <br/>
4) It's done, just run train_DFS.py. 

## **Poster:** <br/>
![Poster](Poster.png) 
