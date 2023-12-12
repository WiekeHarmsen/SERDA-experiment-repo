# SERDA-experiments-repo
Scripts to perform ASR experiments. This consists of:
1. Reading annotations from TextGrid files
2. Reading ASR results from json files
3. Aligning ASR results and manual annotations


## Preparation
The scripts in this repo are dependent on two packages: `dartastla` and the prompt files.
Before running these scripts, first install the local package `dartastla`.

1. Clone the repository 
    
    https://github.com/WiekeHarmsen/dartastla

2. Navigate in terminal to repository

    `cd /vol/tensusers5/wharmsen/dartastla`

3. Locally install the package: 
    
    `pip install -e .`

4. Check if the package is added to your path: 

    `python -c "import sys; print(sys.path)"`

In addition, these scripts are dependent on the prompt files /vol/tensusers2/wharmsen/SERDA-data/prompts/(words,story)_{1,2,3}-wordIDX.csv

## Run uber.sh

Set the correct input variables in `uber.sh`.

Run uber.sh:

    nohup time ./uber.sh &


## Description of the scripts

### 01-stories-align-prompt-whispert.py
This script takes as input one ASR_result (whisperT, .json) and corresponding prompt file (.prompt).
This script reads the files, extracts the relevant information, aligns the prompt and ASR transcription and outputs this.
The relevant information from the prompt file is only the reference transcription (what the child should read).
The relevant information from the ASR result file is the 'segments' property. 
This is a dictionary with as value an object with the following word properties: label, start_time, end_time and confidence score.

This script is an improved version of /vol/tensusers5/wharmsen/ASTLA/astla-round1/2-story2file-info.ipynb
and /vol/tensusers5/wharmsen/ASTLA/astla-round1/4-add-story-conf-info.ipynb

OUTPUT: 
A directory with one or multiple .csv files with an alignment of whisper output and prompt.

Example:

    promptID    aligned_asrTrans    reversed_aligned_asrTrans   correct confidence  startTime   endTimes
    0-0-Bang    *a*l                *als                        False   0.0         0.0         0.0


### 02-tg2dict.py
This script converts a .TextGrid file with ASTLA manual annotations to a dictionary with promptIDs as key and the annotations as value.

### 03-dict2csv.py
This script gets as input a directory with .json files (output from 02-tg2dict.py) and computes from these files the STUDENT x PROMPT_ID dataframes.
These are exported as CSV files.