## Prerequisites
* [Python >= 3.7.9](https://www.python.org/downloads/release/python-379/)
* [OpenCV](https://opencv.org)
* [NumPy](https://numpy.org)
* [MoviePy](https://pypi.org/project/moviepy/)

## Preparation
Store the query image and target image in the folder: input/videos/<folder_name>
Query image name: mario.png
Target image name: mario.mp4

## Usage
```sh
python videoAnalysis.py -d <input_folder_name>
```
example:
```sh
python videoAnalysis.py -d 1
```

## Return Value
The outputs will be saved automatically in the folder: output/
