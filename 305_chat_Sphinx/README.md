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

Training data for the neural net is in the [bAbi facebook dataset](https://github.com/dgandle/305chatbox/tree/master/Data\ Format) format.  Edit the jupyter notebook to update the file location of the dataset:
'''
train_stories = get_stories(open('happy_sad_train.txt'))
test_stories = get_stories(open('happy_sad_train.txt'))
'''

To add a language model and a dictionary for the speech recognizer edit `__init__.py` in `speech recognizer`

'''
language_model_file = os.path.join(language_directory, "3214.lm")
phoneme_dictionary_file = os.path.join(language_directory, "3214.dic")
'''
and place the `.dic` and `.lm` file in `pocketsphinx-data\en-US`.



