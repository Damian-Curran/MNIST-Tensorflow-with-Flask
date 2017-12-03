# MNIST-Tensorflow-with-Flask

In this repository we will be taking a look at the MNIST data set and investigating its different aspects while using it to train a model in Tensorflow and launching it as a webapp.

TensorFlow is a Python library for fast numerical computing created and released by Google. 
It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.

## MNIST Data Set

There are 4 different files. Two training and two test files, both are layed out in the same way.

The Image files are comprised of initially 16 bytes split into 4 parts of 4 bytes each. 
Theses parts are:
* Magic Number
* Number of Images
* Number of Rows
* Number of Columns

The rest of the Image file holds the bytes for 60,000(10,000 for test) Images.

The Label files consist of 8 bytes which spplit into two parts.
Theses parts are:
* Magic Number
* Number of Labels

The rest of the Label file holds the bytes for 60,000(10,000 for test) Labels.

## Installing needed technologies

You'll will first need to install Python, this can be done by following this link and its instructions, install Python 3 [Pyton Install](https://www.howtogeek.com/197947/how-to-install-python-on-windows/)
Make sure Python is in your enviroment variables path.
To check if Python installed properly, open your cmd and type "Python".

You should then install the Tensorflow library, this is done by opening your cmd and typing "pip3 install --upgrade tensorflow" for the CPU-only version and "pip3 install --upgrade tensorflow-gpu" for the GPU version.

You will need the numpy library too, to do so, open your cmd and enter "pip install numpy"

If you want to run the Jupyter Notebook by its self you'll also have to install it using: pip3 install jupyter

Installing FLask: pip install Flask

Installing PIL: sudo pip install pillow

## Using this repository

git clone https://github.com/Damian404/MNIST-Tensorflow-with-Flask.git

open your command console, navigate into the folder and type these commands:
* set FLASK_APP=app.py
* flask run
