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

## **Poster:** <br/>
![Poster](Poster.png)
