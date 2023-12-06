import pandas as pd
import glob
import os
import json
import re
from unidecode import unidecode

# Custom packages
import alignment_adagt.string_manipulations as strman
import alignment_adagt.adagt_preprocess as prep
import alignment_adagt as adagt

import whisper_utils as whutil

# nohup time python ./extract-story-info.py &

storyDataDir = '/vol/tensusers2/wharmsen/SERDA/round1/'

# Audio
audioPathStories = os.path.join(storyDataDir, 'audio/stories')

# Prompt - Get transcription, normalize transcription
promptsPathStories = os.path.join(storyDataDir, 'prompts/stories')

# ASR - Get transcription, normalize transcription (kan op zelfde manier als woorden denk ik)
asrPathStories = os.path.join(
    storyDataDir, 'asr/stories/serda-whispert-story-prompts')


def readPromptFile(basename):
    # Read one .prompt file
    promptPath = os.path.join(
        promptsPathStories, basename + '.prompt')

    with open(promptPath, 'r') as f:
        promptRaw = f.read().replace('\n', ' ')

    prompt = strman.normalizeText(promptRaw)
    return prompt


def readAsrResult(basename):

    # Read one .json asrresult file
    asrResultStory = os.path.join(
        asrPathStories, basename + '.json')

    whisperToutput = whutil.whisperTOutputJsonToDict(asrResultStory)
    whisperTWordInfo = whutil.extractWordInfoFromWhisperTOutput(whisperToutput)

    asrTranscriptionRaw = " ".join(list(whisperTWordInfo.keys()))
    asrTranscription = strman.normalizeText(asrTranscriptionRaw)

    return asrTranscription


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


def main():
    print('analysis started')

    storiesAudioFileList = glob.glob(os.path.join(audioPathStories, '*.wav'))

    for audioFile in storiesAudioFileList:

        basename = os.path.basename(audioFile).replace('.wav', '')
        print(basename)

        if not os.path.exists('/vol/tensusers2/wharmsen/diagnostics_SERDA/study2.1/word-level/stories/' + basename + '.csv'):

            prompt = readPromptFile(basename)
            try:
                asrTranscription = readAsrResult(basename)

                max_length = 50
                promptList = []
                asrTransList = []
                target_text = prompt
                original_text = asrTranscription
                original_space_idx = asrTranscription.find(" ", max_length)

                promptList, asrTransList, target_text, original_text, original_space_idx, max_length = makeSplit(
                    promptList, asrTransList, target_text, original_text, original_space_idx, max_length)

                promptAlignPartsList = []

                for idx, promptPart in enumerate(promptList):
                    asrTransPart = asrTransList[idx]

                    promptAlignPartDF = adagt.two_way_alignment(
                        promptPart, asrTransPart)
                    promptAlignPartsList.append(promptAlignPartDF)

                promptAlignDF = pd.concat(promptAlignPartsList)
                promptAlignDF.to_csv(
                    '/vol/tensusers2/wharmsen/diagnostics_SERDA/study2.1/word-level/stories/' + basename + '.csv')

            except:
                with open('/vol/tensusers2/wharmsen/diagnostics_SERDA/study2.1/word-level/stories/logs.txt', 'a') as f:
                    f.write(basename)

    print('END OF SCRIPT REACHED')


if __name__ == "__main__":
    main()
