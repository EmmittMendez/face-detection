# face-detection

The Python version for the project is 3.10, and to run the code, it's necessary to enter the following recommended values in the terminal:

cam.py
python cam.py --face cascades/haarcascade_frontalface_default.xml --video video/china.mp4

face-processing.py
python face-processing.py -i faces/face_0.png -ocorrected_faces/facec_0.png -k 255 -g 3 -b 0.005 -a 0.005 -m 0 -n 255 -t 0 -l 3