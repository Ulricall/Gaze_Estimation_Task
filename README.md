# Gaze Estimation
A gaze estimation project for AI2612.

## Task1: Gaze Estimation
Data preparation: put MPIIGazeDataset into ./input(which does not exist at the beginning)

Then run task_1_MPII.py, the code does both training and testing, and it will output output_1.txt, which is the result of testing on p10-p14.

## Task2: Gaze domain adaptation
We calculate the gaze error between MPIIFaceGaze and ColumbiaGaze. 

Put ColumbiaGazeDataSet into ./input

Firstly, Use the work in https://github.com/ageitgey/face_recognition to preprocess the pictures in ColumbiaGaze to cut out faces from the pictures.

Install face_recognition following the above github link, and then run scripts/getlabelColumlia.py and scripts/preprocess.py to get the label of ColumbiaGazeDataSet and overwrite original pictures with cut pictures.

After that run task2_MPII.py which trains on MPII and tests on Columbia and task2_Columbia.py which trains on Columbia and tests on MPII, the outputs are separately 
