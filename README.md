# Image Processing for crack detection of metal material
Because of reflection, the crack on the metal material is hard to detect, in this project, we try to use customize filter to average the grayscale of image to increase the correctness of detection
## Tool
- Python 3.9.0
- OpenCV
- patchify
- scipy 1.13.1
## Presentation
In normal crack detection of OpenCV,<br>
![labelResult_cutting5](https://github.com/user-attachments/assets/dc770200-4aea-466e-8c80-b9d01db38b1d)<br>
the result only detect few cracks has grayscale lower than average grayscale values
and ignore some cracks surrounded by pixels have higher grayscale values.

To filter images in different regions separately to improve detection accuracy, we customize a filter method. First, divide image into few 100*100 size patches, 
and calculate each patch`s entropy by using scipy`s fft(Fast Fourier Transform) function and finally get the average entropy(E_a), next we focus on those patches which entropy is higher than E_a and ladel the crack. The result of our method showed below:<br>
![labelResult_fft_cutting5](https://github.com/user-attachments/assets/f132eb6b-544b-479f-b4d8-4d32fa64b1df)<br>
can observe that although still affected by noise and margin of image, the cracks is detected more accurated.
