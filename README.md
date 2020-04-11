Human head pose estimation using Keras over TensorFlow
======================================================
![detections](https://github.com/rafabs97/headpose_final/blob/master/sample_detecions.png)

Work by Rafael Berral-Soler, directed by Manuel J. Marin-Jimenez and Francisco J. Madrid-Cuevas. 
The pose estimator models developed on this project are based on the architectures on the following article:

```
M. Patacchiola y A. Cangelosi. Head pose estimation in the wild using convolutional neural networks and adaptive gradient methods. Pattern Recognition, 71, June 2017.
```

**Old** code for this project can be downloaded [here](https://www.dropbox.com/s/7s4tpcm3jx4ke33/headpose_final.zip?dl=1) (only for archiving reasons).

## Notes
* In order to use this model, you **MUST** import ```models/keras_ssd512.py```, ```bounding_box_utils``` and ```keras_layers``` from the Keras implementation of Single-Shot Multibox Detector by Pierluigi Ferrari found [here](https://github.com/pierluigiferrari/ssd_keras). These files must be placed in the root of the cloned repository directory.
* The original MatConvNet head detector model can be obtained [here](https://github.com/AVAuco/ssd_people), work by Pablo Medina-Suarez and Manuel J. Marin-Jimenez. It must be placed on the ``models/`` directory on the root of the cloned repository directory, and converted by running the script ``convert_ssd_512.py``.
* SSD model converter can also be obtained as a standalone software [here](https://github.com/AVAuco/ssd_people_keras).
* In order to use the model and code included in this repository, it may be useful to update your PYTHONPATH. Provided you cloned this repository at ~/libs/headpose_final:

  ```
  export PYTHONPATH=$PYTHONPATH:~/libs/headpose_final/:~/libs/headpose_final/models/
  ```
  
## Software requirements
* OS: Ubuntu 16.04 or later (and derivatives), Windows 7 or later, macOS 10.12.6 or later (no GPU support).
* Python: Version 3.7.6.
* NVIDIA Software: GPU Drivers (410.x or later), CUDA (10.0) and cuDNN (7.4.1 or later for CUDA 10.0).
* TensorFlow: with or without GPU support, versión 1.14.0.
* Keras: Version 2.2.4.
* OpenCV: Version 4.1.0.25.
* Other Python libraries: glob2 (0.7), numpy (1.18.1), pandas (1.0.1), pillow (7.0.0), scikit-learn (0.22.2).

## Quick Start guide
* Clone this repository to a directory of your choice.

  ```
  git clone https://github.com/rafabs97/headpose_final <directory>
  ```
* Install all required libraries. You can do this manually or using the requirements file included in this repository (**WARNING: Potentially outdated**).

  ```
  pip install -r <directory>/requirements.txt
  ```
* Place ```models/keras_ssd512.py```, ```bounding_box_utils``` and ```keras_layers``` from [here](https://github.com/pierluigiferrari/ssd_keras) at the root of the cloned repository directory.
* Place ```models/head-detector.mat``` from [here](https://github.com/AVAuco/ssd_people) on the ``models/`` directory on the root of the cloned repository directory.
* Run the script  ``convert_ssd_512.py``.
* Run the script ``demo_image.py``. The output should match the demo output at the top of this page.

## Comparison

We´ve added the code we used to compare our model with the one by Patacchiola et al. that can be found [here](https://github.com/mpatacchiola/deepgaze). In order to use it:

* Place ```deepgaze/face_detection.py``` and ```deepgaze/head_pose_estimation.py```from the previous repository on the ``comparison/`` directory on the root of the cloned repository directory.
  * You may need to replace any reference to cv2.cv.CV_HAAR_SCALE_IMAGE to cv2.CASCADE_SCALE_IMAGE in ```face_detection.py```.
  * You can remove the dlib verification and the entire PnpHeadPoseEstimator from ```head_pose_estimation.py```, as it won't be used in     the comparison.
* Place ```etc/tensorflow/head_pose/tilt/``` and ```etc/tensorflow/head_pose/pan/``` from the same repository on the ``comparison/models/`` directory of the cloned repository directory (create the directory if needed).
* Place ```etc/xml/haarcascade_frontalface_alt.xml``` and ```etc/xml/haarcascade_profileface.xml``` from the same repository on the ``comparison/models`` directory of the cloned repository directory.

The functions of the scripts contained are the following: 

* Processed dataset for the comparison have been obtained using the script ``clean_aflw.py`` on the ``comparison/`` directory (scripts ``clean_pointing04.py`` and ``clean_all.py`` can be used to process the Pointing'04 dataset and to obtain the combined dataset in the same way we processed the data for our training, but using as head detector the Viola-Jones detector).
* Script ``match_datasets.py`` can be used to naïvely extract a subset from a dataset appearing on a second dataset, based only on the tilt and pan head pose values (we have used it to match pictures from the AFLW dataset, as it should be rare to find repeating values in that dataset). 
* We used script ``test_on_dataset.py`` to get the mean error of the models by
Patacchiola et al. over the dataset obtained using the previous scripts.

## Video demo

A video processed with this software can be found [here](https://youtu.be/sMPB2xXe8eY).

## Acknowledgements
Picture used in demo by [Ross Broadstock](https://www.flickr.com/people/figurepainting/). Licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license.

Original video used in demo by [Bailey Fisher](https://www.youtube.com/channel/UCFBrplvSu0C16ThC11_OoCg).
