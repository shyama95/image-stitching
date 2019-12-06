# Basic Image Stitching application
## Abstract
Image stitching refers to the process of combining multiple images of the same scene, with overlapping regions, to produce an image of larger resolution. This application implement a feature based image stitching application using python. The application uses SIFT algorithm for feature detection, FLANN for feature matching and RANSAC for homography computation of matched features. It implements image stitching of two images when left and right images were known which is invariant to rotation, scale changes, warping, moderate noise levels and exposure changes.

## Dependencies
- python v3
- PyQt4
- python libraries : opencv-contrib (v3.4.2.16), numpy, matplotlib, scipy (v1.1.0)
## Instructions to run
~~~~
python3 main.py
~~~~
## Results
A screenshot of the application is given below.

![Basic Image Stitching Application Screenshot](https://github.com/shyama95/image-stitching/blob/master/images/application-screenshot.png)
  
## References
[1] Gonzalez, Rafael C., and Woods, Richard E. "Digital image processing. 4E" (2017).  
[2] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.  
[3] Lowe, D. G. (1999). Object recognition from local scale-invariant features. In Computer vision, 1999. The proceedings of the seventh IEEE international conference on (Vol. 2, pp. 1150-1157). Ieee.  
[4] Juan, L., Gwun, O. (2009). A comparison of sift, pca-sift and surf. International Journal of Image Processing (IJIP), 3(4), 143-152.  
[5] Adel, E., Elmogy, M., Elbakry, H. (2014). Image stitching based on feature extraction techniques : a survey. International Journal of Computer Applications (0975-8887) Volume.  
[6] http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/  
[7] Image source : https://commons.wikimedia.org/wiki/Category:Historical_images
