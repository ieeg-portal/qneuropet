## Pipeline Script

### Pre-Requisites

The pipeline script assumes you have the following requirements installed:

[OpenCV](https://pypi.org/project/opencv-python/)
[ANTS](http://stnava.github.io/ANTs/)

The following python module requirements:
cv2
nibabel
numpy
pandas
scipy

This has been tested on MacOS 10.15 and Linux openSUSE Leap 15.2.

### How to use the script
The script should be run with a CSV of patient image locations with resection hemisphere ("L" or "R"). The script takes as command line inputs of CSV file location, directory where output images should be saved, and number of cores to use for computation. An example command to run would be:

```
python pipeline.py /path/to/csv_file.csv /path/to/output/dir/ 12
```

The script generates an output HTML called "out.html" in "/path/to/output/dir" as well patient-specific folders with computed PET feature images.

Having trouble with qNeuroPET? [Contact support](http://github.com/ieeg-portal) and we’ll help you sort it out.
