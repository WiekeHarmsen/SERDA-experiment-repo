"""
This file is largely based on SERDA-data/round2/04-preprocess-data.py
"""

import librosa
import os
import glob
import numpy as np
import pandas as pd
import argparse

def getPromptDF(task):

    pathToPromptIdxs = '/vol/tensusers2/wharmsen/SERDA-data/prompts/'

    promptFileName = task + '-wordIDX.csv'
    promptFile = os.path.join(pathToPromptIdxs, promptFileName)

    promptDF = pd.read_csv(promptFile)

    return promptDF

def getSpeakerIDs():
    df =  pd.read_csv('/vol/tensusers2/wharmsen/SERDA-data/prompts/round1_speakerIDs.csv')
    return list(df['round1_speaker_ids'])


"""
This function adds empty dataframes (col=word_ids, row=students) as values to the storyInfoDict
"""
def initializestoryInfoDict(storyInfoDict, uniqueStudents, word_ids, taskStr):

    storyInfoDict['storyAsrTimeDF'.replace('story', taskStr)] = pd.DataFrame(index = uniqueStudents, columns=word_ids)
    storyInfoDict['storyAsrAccuracyDF'.replace('story', taskStr)] = pd.DataFrame(index=uniqueStudents, columns=word_ids)
    storyInfoDict['storyAsrConfidenceDF'.replace('story', taskStr)] = pd.DataFrame(index = uniqueStudents, columns = word_ids)
    storyInfoDict['storyAsrStartSpeakDF'.replace('story', taskStr)] = pd.DataFrame(index = uniqueStudents, columns = word_ids)
    storyInfoDict['storyAsrStopSpeakDF'.replace('story', taskStr)] = pd.DataFrame(index = uniqueStudents, columns = word_ids)

    return storyInfoDict


"""
Extract word-level logs and save to storyInfoDict
"""
def extractWordInfo(logDF, taskID, studentID, storyInfoDict):

    for idx, row in logDF.iterrows():

        # Word information (directly from log)
        promptID = logDF.loc[idx,'promptID']
        accuracy = 1 if logDF.loc[idx,'correct'] == True else 0
        confidence = logDF.loc[idx,'confidence']
        startRecording = logDF.loc[idx,'startTime']
        stopRecording = logDF.loc[idx,'endTimes']

        # Word information (derived)
        pronunciationTime_ms = stopRecording - startRecording

        # Save word information in the right dataframe of the storyInfoDict
        confKey = 'storyAsrConfidenceDF'.replace('story', taskID).replace('_', '')
        storyInfoDict[confKey].loc[studentID, promptID] = confidence
        timeOutputKey = 'storyAsrTimeDF'.replace('story', taskID).replace('_', '')
        storyInfoDict[timeOutputKey].loc[studentID, promptID] = pronunciationTime_ms
        manualCorrectKey = 'storyAsrAccuracyDF'.replace('story', taskID).replace('_', '')
        storyInfoDict[manualCorrectKey].loc[studentID, promptID] = accuracy
        startSpeakKey = 'storyAsrStartSpeakDF'.replace('story', taskID).replace('_', '')
        storyInfoDict[startSpeakKey].loc[studentID, promptID] = startRecording
        stopSpeakKey = 'storyAsrStopSpeakDF'.replace('story', taskID).replace('_', '')
        storyInfoDict[stopSpeakKey].loc[studentID, promptID] = stopRecording

    return storyInfoDict

"""
Extract file-level (= one task by one speaker) information
"""
def extractFileInfo(fileID, taskID, studentID, logDF):

    nrOfWords = len(logDF)

    reviewArray = list(logDF['correct'])
    nrCorrect = reviewArray.count(True)
    nrIncorrect = reviewArray.count(False)

    return [fileID, studentID, taskID, nrOfWords, nrCorrect, nrIncorrect]


"""
Extract word and file level information.
"""
def extract_info_from_logs(story_asr_csv_dir, storyInfoDict):

    # List all log files
    storyCsvFiles = glob.glob(os.path.join(story_asr_csv_dir, '*.csv'))

    # Initialize output matrix
    fileInfoMatrix = []

    for csvFile in storyCsvFiles:

        # Read log file
        logDF = pd.read_csv(csvFile, sep=',')
        fileID = os.path.basename(csvFile).split('-20')[0]

        # File information
        studentID = fileID.split('-')[0]
        taskID = fileID.split('-')[1]

        # Extract file-level logs and save to fileInfoMatrix
        fileInfoMatrix.append(extractFileInfo(fileID, taskID, studentID, logDF))

        # Extract word-level logs and save to storyInfoDict
        storyInfoDict = extractWordInfo(logDF, taskID, studentID, storyInfoDict)

    # Save file-level info as DF
    fileInfoDF = pd.DataFrame(fileInfoMatrix, columns = ['fileID', 'userID', 'taskID', 'nrOfWords', 'nrCorrect', 'nrIncorrect']).sort_values(['fileID'])

    return fileInfoDF, storyInfoDict

"""
Export file level information: one dataframe to one tsv file
"""
def export_file_level_info(fileInfoDF, outputPath, name_output_file):

    #Export file level measures
    fileLevelOutputFile = os.path.join(outputPath, name_output_file)
    fileInfoDF.to_csv(fileLevelOutputFile, sep='\t')

"""
Export word level information, 5 dataframes for each task
"""
def export_word_level_info(wordInfoDict, outputPath):
    for key in wordInfoDict.keys():
        wordInfoDict[key].to_csv(os.path.join(outputPath, key + '.tsv'), sep='\t')

def run(args):

    story_asr_csv_dir = args.csv_dir
    output_dir = args.output_dir

    storyFileList = glob.glob(os.path.join(story_asr_csv_dir, '*.csv'))

    prompt_ids_story1 = getPromptDF('story_1')['prompt_id']
    prompt_ids_story2 = getPromptDF('story_2')['prompt_id']
    prompt_ids_story3 = getPromptDF('story_3')['prompt_id']

    # Initialize output DFs (studentIDs x promptIDs)
    storyInfoDict = {}
    storyInfoDict = initializestoryInfoDict(storyInfoDict, getSpeakerIDs(), prompt_ids_story1, 'story1')
    storyInfoDict = initializestoryInfoDict(storyInfoDict, getSpeakerIDs(), prompt_ids_story2, 'story2')
    storyInfoDict = initializestoryInfoDict(storyInfoDict, getSpeakerIDs(), prompt_ids_story3, 'words3')

    # Extract file and word level measures from logs
    fileInfoDF, storyInfoDict = extract_info_from_logs(story_asr_csv_dir, storyInfoDict)

    # Write the file-level and word-level measures
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    export_file_level_info(fileInfoDF, output_dir, 'words-asr-filelevel-data.tsv')
    export_word_level_info(storyInfoDict, output_dir)

    print("Finish script 04: Preprocess word logs, see output in:", output_dir)

        
def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--csv_dir", type=str, help = "Dir with csv files that contain for each prompt a cor/inc score. This is the output of the previous script.")
    parser.add_argument("--output_dir", type=str, help = "Output directory - asr log measures")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()