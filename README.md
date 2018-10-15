# imagepredict_mnist

This is an application used to predict a handwritting number(from 0 to 9) with scale of 28 pix *28 pix.
After pull the docker container you may start the predicting with running the docker container and cassandra container.
You should use curl options to upload your images and you can also view data through the cassandra database, including what time the image is uploaded, names of images(names are changed for security consideration) and predictions.
#the flowing is some tips#
"MnistAPP.py" is the main application of this project based on python3.6.
MnistAPP.py uses a mnist model (trained for 20000 times) named "model2".
*Uploading images is required when using this app and uploaded images will be stored in a folder named "uploaded_images" which should be in the same directory as MnistAPP.py.
*"uploaded_images" folder is not in this branch as you see; erros will be reported without this folder.
The latest cassandra docker container is required.
