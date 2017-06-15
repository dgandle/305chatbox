# 305chatbox
###### A GRE Vocab Builder / Alexa Chatbox for CSC 305

Group Name: 305 Chat Box
Group Members: Timothy Leonard, Doug Gandle, Ross Bohensky, Jeffrey Oladapo


---


## Using pocketsphinx to process live speech using keyword search

Paste `ps_recognizer.py` into a jupyter notebook and copy kws_thresh.txt into the directory where the notebook is launched.  Speak into your microphone the phrases included in `kws_thresh.txt`.

[Download pocketsphinx here](https://pypi.python.org/pypi/pocketsphinx) 

### Keyword list threshold file

The key word list does the work of limiting the vocabulary to only key phrases included in the list <b>AND</b> allows you to set the threshold for each phrase.  The keyword list with thresholds looks like this:

```
does happy/1e-25/
mean sad/1e-25/
scratch that/1e-20/
validate/1e-20/
exit/1e+20/
```

You can change the phrases and the thresholds.  It is recommended that longer phrases like "does happy mean sad" are broken into shorter phrases.  You have to experiment with thresholds to find out what works.

[Click for more on keyword lists](https://cmusphinx.github.io/wiki/tutoriallm/#keyword-lists)

#### [CMUSphinx](https://cmusphinx.github.io/)
