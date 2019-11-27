# Video-Stabilization

The objective of this project was to replicate YouTube’s video stabilisation algorithm. 
The main characteristic that makes it different from any other video stabilization algorithm
is that it transforms the camera’s path into only constant, linear and parabolic segments. A
constant would represent something filmed using a tripod, a linear segment would represent
something filmed using a dolly or a pan, and the parabolic segments create the transitions. 


Setup
-----
To use the algorithm the user has to: 
1. Place the main.py and the videostabilisation.py files in the same directory as the input
   video
2. Replace the VIDEO_NAME, VIDEO_EXTENSION, and the CROP_PERCENTAGE variables by the input 
   video name, its extension and the amount of cropping to add, respectively
3. Download Python, Numpy and OpenCV on your computer
4. On your command-line tool, navigate to the directory and type: python3 main.py

Code language
-------------
Python 

List of code files
------------------
videostabilization.py
main.py

Link to inputs/outputs
----------------------
https://drive.google.com/open?id=1TAnLP9_G8qSupPxeShclAFcjdkj4b1xh
