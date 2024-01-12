"""
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
"""

import pandas as pd
import glob
import os
import json
import re
from unidecode import unidecode
import argparse
from datetime import datetime

# Local package 'dartastla'
# First install this package by
# cd /vol/tensusers5/wharmsen/dartastla
# pip install -e .
# python -c "import sys; print(sys.path)"
import alignment_adagt.string_manipulations as strman
import alignment_adagt.adagt_preprocess as prep
import alignment_adagt as adagt
import whisper_utils as whutil

# nohup time python ./02-stories-align-prompt-whispert.py &

"""
This function reads one .prompt file and normalizes the text (trim spaces, remove accents, remove punctuation, remove digits)
"""
def readPromptFile(path_to_prompt_file):

    with open(path_to_prompt_file, 'r') as f:
        promptRaw = f.read().replace('\n', ' ')

    prompt = strman.normalizeText(promptRaw)
    return prompt

"""
This function reads one json file with an WhisperT AsrResult.
The asrTranscription is normalized (trim spaces, remove accents, remove punctuation, remove digits)
"""
def readAsrResult(asrResultFile):

    whisperToutput = whutil.whisperTOutputJsonToDict(asrResultFile)
    wordDictIdxBased, wordDictLabelBased = whutil.extractWordInfoFromWhisperTOutputWithIDs(whisperToutput)
    
    asrTranscriptionRaw = " ".join(list(wordDictLabelBased.keys()))
    asrTranscription = strman.normalizeText(asrTranscriptionRaw)

    return asrTranscription, wordDictIdxBased


"""
This is a recursive function. The dynamic alignment algorithm ADAGT doesn't work on long texts. 
Therefore, we find a piece of text that occurs in both transcriptions, and split the long text into shorter texts at this piece of text.
This is a recursive function to make this splitting happen.
The properties targetPartsList and origPartsList are in the end used for further analysis.

targetPartsList     string[]:   The target_text splitted in utterances
origPartsList       string[]:   The original_text splitted in utterances
target_text         string:     The complete prompt (can be unlimited words).
original_text       string:     The reading of the prompt as recognized by the ASR.
original_space_idx  int:    The idx at which a split can be made
max_length          int:    Length of utterance.
"""
def makeSplit(targetPartsList, origPartsList, target_text, original_text, original_space_idx, max_length):

    # Get slice to split on
    original_slice = original_text[original_space_idx-3:original_space_idx+3]

    # Find slice in target text
    target_slice_idx = target_text.find(
        original_slice)

    # If slice is found in target text
    if (target_slice_idx != -1 and len(original_slice) != 0):

        # Split the target_text and original_text on the space in the overlapping slice
        # Add the first part to the partLists
        target_space_idx = target_slice_idx+3
        targetPartsList.append(target_text[:target_space_idx])
        origPartsList.append(original_text[:original_space_idx])

        # Remove the first part from the target_text and original_text
        target_text = target_text[target_space_idx+1:]
        original_text = original_text[original_space_idx+1:]

        if (len(original_text) < 80 or len(target_text) < 80):
            # End of file reached
            targetPartsList.append(target_text)
            origPartsList.append(original_text)
            target_text = ''
            original_text = ''
            idx_of_next_space = -1

            return targetPartsList, origPartsList, target_text, original_text, idx_of_next_space, max_length
        else:
            # reset around_idx
            new_space_idx = original_text.find(" ", max_length)

            # make next split
            return makeSplit(targetPartsList, origPartsList, target_text, original_text, new_space_idx, max_length)

    # If slice is not found in target text
    else:
        # reset around_idx to next space
        idx_of_next_space = original_text.find(" ", original_space_idx+1)
        if (idx_of_next_space == -1):
            # End of file reached
            targetPartsList.append(target_text)
            origPartsList.append(original_text)
            return targetPartsList, origPartsList, target_text, original_text, idx_of_next_space, max_length
        else:
            return makeSplit(targetPartsList, origPartsList, target_text, original_text, idx_of_next_space, max_length)

"""
Function that checks whether the three input files exist, if not: print error message.
"""
def checkIfFilesExist(audioFile, asrResultFile, promptFile):

    for file in [audioFile, asrResultFile, promptFile]:
        if not os.path.exists(audioFile):
            print(file, 'does not exist.')

"""
Function that reads the input files, calls the alignment algorithm and prints the output.

asrTranscription    string
prompt              string
"""
def alignOneFile(asrTranscription, prompt):

    # Split prompt and whisperT transcription file in smaller utterances.
    max_length = 50
    promptList = []
    asrTransList = []
    target_text = prompt
    original_text = asrTranscription
    original_space_idx = asrTranscription.find(" ", max_length)
    promptList, asrTransList, target_text, original_text, original_space_idx, max_length = makeSplit(
        promptList, asrTransList, target_text, original_text, original_space_idx, max_length)

    # Apply ADAGT two-way alignment to align the prompt with the whisperT transcription
    promptAlignPartsList = []
    for idx, promptPart in enumerate(promptList):
        asrTransPart = asrTransList[idx]
        promptAlignPartDF = adagt.two_way_alignment(
            promptPart, asrTransPart)
        promptAlignPartsList.append(promptAlignPartDF)
    promptAlignDF = pd.concat(promptAlignPartsList)

    return promptAlignDF


"""
dictValue           {label: string, start_time: float, end_time: float, confidence: float} object
asrResultWordsdict  dict:   The key is the index, the values are {label: string, start_time: float, end_time: float, confidence: float} objects. 
                            This dict contains all recognized word segments from the ASR result.
"""
def getIndexOfDictValue(dictValue, asrResultWordsDict):
    return [x[0] for x in asrResultWordsDict.items() if x[1] == dictValue][0]



"""
This function returns a list of objects that have the specified target_label as label.

label               string: One prompt word
asrResultWordsdict  dict:   The key is the index, the values are {label: string, start_time: float, end_time: float, confidence: float} objects. 
                            This dict contains all recognized word segments from the ASR result.

List of {label: string, start_time: float, end_time: float, confidence: float} objects
"""
def searchAllDictValuesWithLabel(target_label, asrResultWordsDict):
    return [val for val in asrResultWordsDict.values() if val['label'] == target_label]


"""
This function searches in asrResultWordsDict to the values that has as label "prompt". If there are multiple options, it chooses the one with the dict index directly

prompt              string: One prompt word
asrResultWordsDict  dict:   The key is the index, the values are {label: string, start_time: float, end_time: float, confidence: float} objects. 
                            This dict contains all recognized word segments from the ASR result.
indexThreshold      int:    Select the first value that has an the 
"""
def searchCorrespondingConfidence(prompt, asrResultWordsDict, indexThreshold):

    # Make list of all dict values with prompt as target_label.
    allDictValues = searchAllDictValuesWithLabel(prompt, asrResultWordsDict)

    # Get indexes in asrResultsWordsDict of this selection of dict values
    allDictIdxs = [getIndexOfDictValue(
        value, asrResultWordsDict) for value in allDictValues]

    # Select one of these dict values. Use the indexThreshold for that.
    currentDictIdx = -1
    for i, idxItem in enumerate(allDictIdxs):
        if (idxItem > indexThreshold):
            currentDictValue = allDictValues[i]
            currentDictIdx = idxItem
            break

    if (currentDictIdx == -1):
        # Means that prompt is part of recognized word (e.g prompt=rilt, recognized=trilt)
        confidence = 999
        start = 999
        end = 999
    else:
        confidence = asrResultWordsDict[currentDictIdx]['confidence']
        start = asrResultWordsDict[currentDictIdx]['start']
        end = asrResultWordsDict[currentDictIdx]['end']

    return currentDictIdx, {'confidence': confidence, 'start': start, 'end': end}

def addConfidenceScores(promptAlignDF, asrTranscription):
    confidence = []
    startTimes = []
    endTimes = []
    asrResultWordsThreshold = -1

    for idx, row in promptAlignDF.iterrows():
        prompt = idx
        aligned_asrTrans = row['aligned_asrTrans']
        reversed_aligned_asrTrans = row['reversed_aligned_asrTrans']
        correct = row['correct']

        if (correct):
            # Search corresponding confidence score
            asrResultWordsIndex, confStartEndResults = searchCorrespondingConfidence(
                prompt, asrTranscription, asrResultWordsThreshold)

            confidence.append(confStartEndResults['confidence'])
            startTimes.append(confStartEndResults['start'])
            endTimes.append(confStartEndResults['end'])
            asrResultWordsThreshold = asrResultWordsIndex

        else:
            confidence.append(0)
            startTimes.append(0)
            endTimes.append(0)


    promptAlignDF['confidence'] = confidence
    promptAlignDF['startTime'] = startTimes
    promptAlignDF['endTimes'] = endTimes

    return promptAlignDF

def getPromptIdxs(basename):

    pathToPromptIdxs = '/vol/tensusers2/wharmsen/SERDA-data/prompts/'

    task = basename.split('-')[1]
    taskType = task.split('_')[0]
    taskNr = task.split('_')[1]

    promptFileName = task + '-wordIDX.csv'
    promptFile = os.path.join(pathToPromptIdxs, promptFileName)

    promptDF = pd.read_csv(promptFile)

    return list(promptDF['prompt_id'])


def alignWithConfidenceScores(audioFile, asrResultFile, promptFile, outputDir, basename):
    
    checkIfFilesExist(audioFile, asrResultFile, promptFile)

    # try:
    # Read prompt file
    prompt = readPromptFile(promptFile)

    # Read whisperT transcription file
    asrTranscription, asrWordInfoDict = readAsrResult(asrResultFile)

    # Align prompt and AsrResult transcription
    promptAlignDF = alignOneFile(asrTranscription, prompt)

    # Add confidence scores with AsrResult
    promptAlignConfDF = addConfidenceScores(promptAlignDF, asrWordInfoDict)

    # Add promptIDs
    promptAlignConfDF['promptID'] = getPromptIdxs(basename)

    # Save the csv output file with the alignment in the output_dir.
    promptAlignConfDF.set_index('promptID').to_csv(os.path.join(outputDir, basename + '.csv'))

    superDirOfOutputDir = os.path.dirname(os.path.dirname(outputDir))
    with open(os.path.join(superDirOfOutputDir, 'asr-transcriptions.tsv'), 'a') as f:
        f.write(basename+ '\t'+ asrTranscription+'\n')
     

def run(args):

    # Analysis type can be either 'file' or 'dir', the inputs and analysis depend on the analysis type.
    analysisType = args.analysis_type

    # Read output directory and create it if it doesn't exist yet.
    outputDir = args.output_dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if(analysisType == 'file'):
        audioFile = args.input_audio
        asrResultFile = args.input_asr_result
        promptFile = args.input_prompt
        basename = os.path.basename(audioFile).replace('.wav', '') 

        alignWithConfidenceScores(audioFile, asrResultFile, promptFile, outputDir, basename)
    
    elif(analysisType == 'dir'):
        audioDir = args.input_audio_dir
        asrResultDir = args.input_asr_dir
        promptDir = args.input_prompt_dir
        
        # List all audio files
        audioFileList = glob.glob(os.path.join(audioDir, '*.wav'))

        # Iterate over each audio file, select the corresponding asrResult and prompt, align the two, save csv output in outputDir
        for idx, audioFile in enumerate(audioFileList):
            
            basename = os.path.basename(audioFile).replace('.wav', '')
            task = basename.split('-')[1]    
            
            promptFile = os.path.join(promptDir, task + '.prompt')
            asrResultFile = os.path.join(asrResultDir, basename + '.json')

            alignWithConfidenceScores(audioFile, asrResultFile, promptFile, outputDir, basename)

            if (idx+1)%10==0:
                print(datetime.now(), ':', idx, 'of', len(audioFileList), 'story files processed.')
        
        print("Script 01 completed: The prompts of all stories are aligned with the ASR results.")
            

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--analysis_type", type=str, help = "Either 'file' 'dir', depends on whether file or dir of files is analyzed.")
    parser.add_argument("--output_dir", type=str, help = "Output directory where csv file with alignment between whisperT output and prompt are saved.")

    #   In case of file:
    parser.add_argument("--input_audio", type=str, help = "Wav audio file of story task")
    parser.add_argument("--input_asr_result", type=str, help = "JSON file with WhisperT transcription of wav input audio")
    parser.add_argument("--input_prompt", type=str, help = "A .prompt file with the prompt of the audio file")

    #   In case of dir:
    parser.add_argument("--input_audio_dir", type=str, help = "Directory with wav audio files of story tasks")
    parser.add_argument("--input_asr_dir", type=str, help = "Directory with JSON WhisperT AsrResult files corresponding to audio.")
    parser.add_argument("--input_prompt_dir", type=str, help = "Directory with .prompt files containing prompts of audio.") 

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
