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

