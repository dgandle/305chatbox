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

Training data for the neural net is in the [bAbi facebook dataset](https://github.com/dgandle/305chatbox/tree/master/Data_Format) format.  Edit the jupyter notebook to update the file location of the dataset:
```
train_stories = get_stories(open('happy_sad_train.txt'))
test_stories = get_stories(open('happy_sad_train.txt'))
```

## Building a Language and Dictionary File

CMU sphinx offers a [web tool](http://www.speech.cs.cmu.edu/tools/lmtool-new.html) to build the language model and the dictioanry.  Upload a text file that includes sentences using the complete vocabulary that can be found in the data set as well as any command that you want pocket sphinx to recognize.  It is not necessary to include every possible sentence, only enough sentences to include all words to be recognized.  The language model automatically creates probabilities to cover all possible combinations of the words in the vocabulary.  The dicationary is automatically created along with the langauge file.


To add a language model and a dictionary for the speech recognizer edit `__init__.py` in `speech recognizer` and place the `.dic` and `.lm` file in `pocketsphinx-data\en-US`.

```
language_model_file = os.path.join(language_directory, "3214.lm")
phoneme_dictionary_file = os.path.join(language_directory, "3214.dic")
```



