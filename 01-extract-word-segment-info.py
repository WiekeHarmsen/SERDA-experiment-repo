import pandas as pd
import glob
import os
import librosa
import json
import re
from unidecode import unidecode
import num2words

# Custom packages
import alignment_adagt.string_manipulations as strman
import alignment_adagt as adagt

"""
cd directory of this script
run: nohup time python ./extract-word-segment-info.py

This script extracts word accuracy scores based on output WhisperTimestamped LargeV2.
Uses both normal and reversed alignment, confidence scores are not taken into account.

The begin/end times are obtained from the log files.

Limitation:
- not corrected for cases when child reads word before clicking

"""


def getFileLists(serdaDir, wordTasks, round):
    fullTaskPath = 'audio/words/full/*.wav'
    fullWordsFileList = glob.glob(
        os.path.join(serdaDir, round, fullTaskPath))

    segmentsTaskPath = 'audio/words/segments/*.wav'
    segmentsWordsFileList = glob.glob(
        os.path.join(serdaDir, round, segmentsTaskPath))
    return fullWordsFileList, segmentsWordsFileList


def whisperTOutputJsonToDict(jsonFile):
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    return data


def extractSegmentInfoFromWhisperTOutput(asrResult):
    wordDict = {}
    for segment in asrResult['segments']:
        words = segment['words']
        for word in words:

            label = word['text']

            # If no dislefluency
            if (label != '[*]'):

                start = word['start']
                end = word['end']
                confidence = word['confidence']

                # Save word information in WordDict
                wordDict[label] = {
                    'confidence': confidence,
                    'start': start,
                    'end': end
                }

    return wordDict


# Create output file
def createOutputFile(outputFilePath):
    with open(outputFilePath, 'w') as f:
        f.write(",".join(['audio', 'promptID', 'promptWord', 'nrOfSegments',
                          'oneOfSegmentsCorrect', 'confOfCorrectSegment', 'durLibrosa', 'logStart', 'logEnd']) + '\n')

# Extract correctness scores for each word segment audio file


def extractAndWriteRelevantData(serdaDir, round, file, wordIDs, outputFile):

    # Get duration (librosa)
    durLibrosa = librosa.get_duration(path=file)

    # Select corresponding ASR result
    asrResultName = os.path.basename(file).replace('.wav', '.json')
    asrResult = os.path.join(
        serdaDir, round, 'asr/words/.serda-whispert-prompts_unfiltered', asrResultName)

    # Extract relevant data (for each word: label, start time, end time, confidence) from AsrResult
    whisperToutput = whisperTOutputJsonToDict(asrResult)
    whisperTsegments = extractSegmentInfoFromWhisperTOutput(whisperToutput)

    # Select corresponding prompt
    promptID = (re.split("[_-]", asrResultName)[3])
    promptWord = wordIDs.loc[int(promptID), 'prompt']

    # Align each segment text with prompt
    whisperTTextSegments = whisperTsegments.keys()
    nrOfSegments = len(whisperTTextSegments)
    oneOfSegmentsCorrect = False
    confOfCorrectSegment = 0
    for idx, segmentText in enumerate(whisperTTextSegments):
        outputDF = adagt.two_way_alignment(
            promptWord, re.sub('([0-9]+)', ' ', strman.removePunctuation(unidecode(segmentText.lower()))))
        confOfSegment = whisperTsegments[list(
            whisperTTextSegments)[idx]]['confidence']
        if (outputDF.loc[promptWord, 'correct'] and confOfSegment > confOfCorrectSegment):
            oneOfSegmentsCorrect = True
            confOfCorrectSegment = confOfSegment

    # Select corresponding log file
    logFileName = re.sub('_[1-3][0-9][0-9]_*\w*', '',
                         asrResultName.replace('.json', '.csv'))
    logFile = pd.read_csv(os.path.join(
        serdaDir, round, 'logs/words/' + logFileName), sep=';').set_index('prompt_id')
    logStart = logFile.loc[int(promptID), 'start_speak'] if ('logstamp' in asrResultName) or (
        'taskstart'not in asrResultName and 'logstamp' not in asrResultName) else 0
    logEnd = logFile.loc[int(promptID), 'stop_speak']

    with open(outputFile, 'a') as f:
        f.write(",".join([str(x) for x in [asrResultName.replace('.json', ''), promptID, promptWord, nrOfSegments,
                oneOfSegmentsCorrect, confOfCorrectSegment, durLibrosa, logStart, logEnd]]) + '\n')


def main():

    # General variables
    serdaDir = '/vol/tensusers2/wharmsen/SERDA/'
    round = 'round1'
    wordTasks = ['words_1', 'words_2', 'words_3']

    # Read prompts word tasks
    wordIDs = pd.read_csv(os.path.join(
        serdaDir, 'docs', 'wordtask-wordIDX.csv')).set_index('prompt_id')

    fullWordsFileList, segmentsWordsFileList = getFileLists(
        serdaDir, wordTasks, round)

    for idx, file in enumerate(segmentsWordsFileList):

        # print progress
        if idx % 50 == 0:
            print(idx, 'of', len(segmentsWordsFileList))

        #
        outputFileName = re.sub(
            '_[1-3][0-9][0-9]_*\w*', '', os.path.basename(file).replace('.wav', '.csv'))
        outputFilePath = '/vol/tensusers2/wharmsen/diagnostics_SERDA/study2.1/word-level/word-segments/'
        outputFile = outputFilePath + outputFileName

        if not os.path.exists(outputFile):
            createOutputFile(outputFile)

        with open(outputFile, 'r') as f:
            presentPromptIDs = [line.split(',')[1] for line in f.readlines()]
            currentPromptID = (re.split("[_-]", os.path.basename(file))[3])

            if currentPromptID not in presentPromptIDs:
                extractAndWriteRelevantData(
                    serdaDir, round, file, wordIDs, outputFile)
            elif currentPromptID == '101' and presentPromptIDs.count('101') == 1:
                extractAndWriteRelevantData(
                    serdaDir, round, file, wordIDs, outputFile)
                presentPromptIDs.append('101')

    print('Check results in:' + outputFilePath)


if __name__ == "__main__":
    main()
