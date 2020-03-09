# Eyes-closing-detector

## Dependencies
OpenCV, dlib

## How to build the program
mkdir build; cd build; cmake ..; cmake --build . 
If cmake cannot find OpenCV, then use: cmake -D OpenCV_DIR="path/to/opencv" .. 

## Notes
The eyes closing detector uses eye aspect ratio (EAR) to determine whether eyes are closing. EAR can be
calculated based on face extracted landmarks given by dlib. This detector work quite ok provided that
you do not wear glasses and close your eyes for 1-2 seconds. 
This code is based on dlib, with added features for eyes closing detector.
