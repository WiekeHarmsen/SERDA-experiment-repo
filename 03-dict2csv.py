""""
This script converts json files with manual annotations (obtained from TextGrids) to accuracy and timing DFs of the format STUDENT x PROMPT_ID.

Input: directory with .json files (output from 02-tg2dict)
Output: directory with .csv files

Dependent on the following files:
'/vol/tensusers2/wharmsen/SERDA-data/prompts/round1_speakerIDs.csv'
'/vol/tensusers2/wharmsen/SERDA-data/prompts/{story1, story2, story3}-wordIDX.csv'
"""

import glob
import os
import json
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

def analyse_accuracy_json(dictFile, outputDF, prompt_ids, speaker_id):
    # Read json object with manual annotations
    with open(dictFile) as f:
        file = json.load(f)

    # extract word correctness info from json object
    for prompt_id in prompt_ids:
        try:
            attempts = file[prompt_id]['attempts']
            if len(attempts)>0:
                annotation = attempts[-1]['correct']
                score = 1 if annotation == 'correct' else 0
            else:
                score = 0
        except:
            # This is happening if the last prompt word is not read by the child.
            score = 0

        outputDF.loc[speaker_id, prompt_id] = score

    return outputDF

def run(args):

    inputDir = args.input_json_dir
    outputDir = args.output_dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    fileList = glob.glob(os.path.join(inputDir, '*.json'))

    # Initialize DFs (studentIDs x promptIDs)
    prompt_ids_story1 = getPromptDF('story_1')['prompt_id']
    story1_accuracy_DF = pd.DataFrame(index=getSpeakerIDs(), columns=prompt_ids_story1)

    prompt_ids_story2 = getPromptDF('story_2')['prompt_id']
    story2_accuracy_DF = pd.DataFrame(index=getSpeakerIDs(), columns=prompt_ids_story2)

    prompt_ids_story3 = getPromptDF('story_3')['prompt_id']
    story3_accuracy_DF = pd.DataFrame(index=getSpeakerIDs(), columns=prompt_ids_story3)

    for dictFile in fileList:

        # Get speakerID
        basename = os.path.basename(dictFile).replace('.json', '')
        speaker_id = basename.split('-')[0]
        task = basename.split('-')[1]
        print('processing', basename, speaker_id, task)

        # Select outputDF (depends on task)
        if(task=='story_1'):
            story1_accuracy_DF = analyse_accuracy_json(dictFile, story1_accuracy_DF, prompt_ids_story1, speaker_id)
        elif(task=='story_2'):
            story2_accuracy_DF = analyse_accuracy_json(dictFile, story2_accuracy_DF, prompt_ids_story2, speaker_id)
        elif(task=='story_3'):
            story3_accuracy_DF = analyse_accuracy_json(dictFile, story3_accuracy_DF, prompt_ids_story3, speaker_id)

    # Export Accuracy DFs
    story1_accuracy_DF.to_csv(os.path.join(outputDir, 'story1_manual_accuracy.csv'))
    story2_accuracy_DF.to_csv(os.path.join(outputDir, 'story2_manual_accuracy.csv'))
    story3_accuracy_DF.to_csv(os.path.join(outputDir, 'story3_manual_accuracy.csv'))

def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--input_json_dir", type=str, help = "A directory with .json files having the output format from 02-tg2dict.py")
    parser.add_argument("--output_dir", type=str, help = "Output directory to save STUDENT x PROMPT_ID csv files.")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
