import os

# you need to have pockect sphinx https://pypi.python.org/pypi/pocketsphinx
#  
# download this in your tensor flow environment if you want to use NN
# 
# run this in a jupyter notebook so you can interpret it on the fly

from pocketsphinx import LiveSpeech, get_model_path

model_path = get_model_path()

# every thing commented out are defaults
# the only thing that matters here is that lm=False
# since this is a key word search
# speech is the decoder
speech = LiveSpeech(
    #verbose=False,
    #sampling_rate=16000,
    #buffer_size=2048,
    #no_search=False,
    #full_utt=False,
    #hmm=os.path.join(model_path, 'en-us'),
    
    lm=False, #####os.path.join(model_path, 'en-us.lm.bin'),
    
    #dic='3850.dic'#os.path.join(model_path, 'cmudict-en-us.dict')
)

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
