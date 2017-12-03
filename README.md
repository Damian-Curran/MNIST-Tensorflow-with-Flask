# MNIST-Tensorflow-with-Flask

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
