# Gaze Estimation
A gaze estimation project for AI2612.

## Task1: Gaze Estimation
Data preparation: put MPIIGazeDataset into ./input(which does not exist at the beginning)

Then run task_1.py, the code does both training and testing, and it will output output_1.txt, which is the result of testing on p10-p14.

## Task2: Gaze domain adaptation
Generalize between MPIIFaceGaze and ColumbiaGaze. 

Firstly, Use the work in https://github.com/ageitgey/face_recognition to preprocess the pictures in ColumbiaGaze to cut out faces from the pictures.

Install face_recognition following the above github link, and then run getlist.py and preprocess.py to overwrite original pictures with cut pictures.
