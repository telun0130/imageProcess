# Image Processing for crack detection of metal material
Because of reflection, the crack on the metal material is hard to detect, in this project, we try to use customize filter to average the grayscale of image to increase the correctness of detection
## Tool
- Python 3.9.0
- OpenCV
- patchify
- scipy 1.13.1
## Presentation
In normal crack detection of OpenCV,
![labelResult_cutting5](https://github.com/user-attachments/assets/dc770200-4aea-466e-8c80-b9d01db38b1d)
the result only detect few cracks has grayscale lower than average grayscale values
and ignore some cracks surrounded by pixels have higher grayscale values.

To filter images in different regions separately to improve detection accuracy, 
