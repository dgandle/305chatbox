# 305chatbox
###### A GRE Vocab Builder / Alexa Chatbox for CSC 305

Group Name: 305 Chat Box
Group Members: Timothy Leonard, Doug Gandle, Ross Bohensky, Jeffrey Oladapo


---


#### [Facebook bAbi tasks format](https://research.fb.com/downloads/babi/)
The file format for each task is as follows:

```
ID text
ID text
ID text
ID question[tab]answer[tab]supporting fact IDS.
...
```
The IDs for a given “story” start at 1 and increase. When the IDs in a file reset back to 1 you can consider the following sentences as a new “story”. Supporting fact IDs only ever reference the sentences within a “story”.

For example:

```
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway         4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway         4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office          11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom        8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra?      bathroom        2
```


