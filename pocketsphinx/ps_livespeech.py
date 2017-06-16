# you need to have pockect sphinx https://pypi.python.org/pypi/pocketsphinx
# download this in your tensor flow environment if you want to use NN
# run this in a jupyter notebook so you can interpret it on the fly

import os
from pocketsphinx import LiveSpeech

# since this is a key word search the only thing that matters here is that lm=False
# everything else is default
# speech is the decoder
speech = LiveSpeech(lm=False)

# Here is where the decoder is set for key word search
# Documentation: https://cmusphinx.github.io/wiki/tutoriallm/
speech.set_kws('keyphrase', 'kws_thresh.txt')
speech.set_search('keyphrase')

# to collect our utterances
word_list = []

# magically the decoder iterates through every perceived utterance
# the keyword 'exit' will break the loop otherwise you have to manually press stop
for phrase in speech:
    if str(phrase) == 'exit': break
    if str(phrase) != '':
        word_list.append(str(phrase))
        print(phrase)
        
print(word_list)
