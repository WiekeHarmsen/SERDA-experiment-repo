import pandas as pd
import glob
import os
import json
import re
from unidecode import unidecode
import argparse
from datetime import datetime

import alignment_adagt.string_manipulations as strman
import alignment_adagt.adagt_preprocess as prep
import alignment_adagt as adagt
import whisper_utils as whutil

# def readAsrResult(asrResultFile):

#     whisperToutput = whutil.whisperTOutputJsonToDict(asrResultFile)
#     wordDictIdxBased, wordDictLabelBased = whutil.extractWordInfoFromWhisperTOutputWithIDs(whisperToutput)
    
#     asrTranscriptionRaw = " ".join(list(wordDictLabelBased.keys()))
#     asrTranscription = strman.normalizeText(asrTranscriptionRaw)

#     return asrTranscription, wordDictIdxBased

def run(args):
    story1Matrix = []
    story2Matrix = []
    story3Matrix = []
    words1Matrix = []
    words2Matrix = []
    words3Matrix = []

    outputDir = args.output_dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    asrResultDir = args.input_asr_dir

    asrResultList = glob.glob(os.path.join(asrResultDir, '*.json'))

    for asrResultFile in asrResultList:
        filename = os.path.basename(asrResultFile)
        studentID = filename.split('-')[0]
        task = filename.split('-')[1]

        whisperToutput = whutil.whisperTOutputJsonToDict(asrResultFile)
        # segment_length_list = [len(whisperToutput['segments'][x]['words']) for x in range(len(whisperToutput['segments']))]
        # nr_of_rec_words = sum(segment_length_list)
        nr_of_rec_words = len(whisperToutput['text'].split(' '))

        if(task == 'story_1'):
            story1Matrix.append([studentID, nr_of_rec_words])
        if(task == 'story_2'):
            story2Matrix.append([studentID, nr_of_rec_words])
        if(task == 'story_3'):
            story3Matrix.append([studentID, nr_of_rec_words])
        if(task == 'words_1'):
            words1Matrix.append([studentID, nr_of_rec_words])
        if(task == 'words_2'):
            words2Matrix.append([studentID, nr_of_rec_words])
        if(task == 'words_3'):
            words3Matrix.append([studentID, nr_of_rec_words])

    task_type = args.task_type
    if(task_type=='stories'):
        pd.DataFrame(story1Matrix, columns=['studentID', 'nrRecWords']).to_csv(os.path.join(outputDir, 'story_1' + '_nr_rec_words.tsv'), index=False)
        pd.DataFrame(story2Matrix, columns=['studentID', 'nrRecWords']).to_csv(os.path.join(outputDir, 'story_2' + '_nr_rec_words.tsv'), index=False)
        pd.DataFrame(story3Matrix, columns=['studentID', 'nrRecWords']).to_csv(os.path.join(outputDir, 'story_3' + '_nr_rec_words.tsv'), index=False)
        print('Created: ', os.path.join(outputDir, 'stories_1' + '_nr_rec_words.tsv'))
    elif(task_type == 'words'):
        pd.DataFrame(words1Matrix, columns=['studentID', 'nrRecWords']).to_csv(os.path.join(outputDir, 'words_1' + '_nr_rec_words.tsv'), index=False)
        pd.DataFrame(words2Matrix, columns=['studentID', 'nrRecWords']).to_csv(os.path.join(outputDir, 'words_2' + '_nr_rec_words.tsv'), index=False)
        pd.DataFrame(words3Matrix, columns=['studentID', 'nrRecWords']).to_csv(os.path.join(outputDir, 'words_3' + '_nr_rec_words.tsv'), index=False)
        print('Created: ', os.path.join(outputDir, 'words_1' + '_nr_rec_words.tsv'))
    

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--model_name", type=str, help = "")
    parser.add_argument("--input_asr_dir", type=str, help = "")
    parser.add_argument("--output_dir", type=str, help = "") 
    parser.add_argument("--task_type", type=str, help = "" )

    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
