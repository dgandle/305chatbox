# 305chatbox
###### A GRE Vocab Builder / Alexa Chatbox for CSC 305

Group Name: 305 Chat Box
Group Members: Timothy Leonard, Doug Gandle, Ross Bohensky, Jeffrey Oladapo


---


### Dependencies

* speech_recognition
* keras
* tensorflow
* numpy
* functools


## Setup

Training data for the neural net is in the [bAbi facebook dataset](https://github.com/dgandle/305chatbox/tree/master/neuralnet) format.  Edit the jupyter notebook to update the file location of the dataset.

To add a language model and a dictionary for the speech recognizer edit `__init__.py` in `speech recognizer` and place the `.dic` and `.lm` file in `pocketsphinx-data\en-US`.

