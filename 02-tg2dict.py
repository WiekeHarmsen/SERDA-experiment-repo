"""
This script reads a .TextGrid (or a directory with .TextGrids) with annotations according to the ASTLA annotation protocol.
It checks for each prompt word whether this occurs in the 'chunks' tier of the TextGrid.
It returns for each input TextGrid a json file.
This json file contains a dictionary with the the prompt_ids as key and the start time, end time, chunk text and chunk attempts as value.
This json file is saved in the outputDir.
"""


import glob
import pandas as pd
import os
import numpy as np
import tgt # https://textgridtools.readthedocs.io/en/stable/api.html
import argparse
import json
from distutils.util import strtobool


"""
This function reads a .tg file and saves it as a dataframe where each row represents one interval.
The columns are the following five properties: tier_name, tier_type, start_time, end_time, text
"""
def read_textgrid_to_dataframe(tg_file):
    
    # Read TextGrid file
    tg = tgt.io.read_textgrid(tg_file, encoding='utf-8', include_empty_intervals=False)

    # Convert TextGrid file to Formatted Table (= df with on each row one interval)
    table = tgt.io.export_to_table(tg, separator=', ')
    formatted_table = [x.split(', ') for x in table.split('\n')]

    tg_df = pd.DataFrame(formatted_table[1:], columns = formatted_table[0])

    # convert start_time and end_time from str to float
    convert_dict = {'start_time': float,
                    'end_time': float,
                    }
 
    tg_df = tg_df.astype(convert_dict)

    return tg_df

# Save relevant info from tier 1: prompts in chunks_df
def initialize_chunks_df(tg_df):
    chunks_df = tg_df[tg_df['tier_name'] == 'chunks']
    chunks_df = chunks_df.drop(['tier_name', 'tier_type'], axis=1)
    return chunks_df

def getAttemptInfo(oneAttemptDF, tier_name, information_type):
    try:
        df = oneAttemptDF[oneAttemptDF['tier_name'] == tier_name].reset_index(drop=True)
        if len(df)==0:
            print('Not all tiers are filled in. We only have info for these tiers:')
            print(oneAttemptDF)
        return df.loc[0, information_type]
    except:
        return ''
    
def getCorrespondingAttempts(attemptsInfoDF, chunk_start, chunk_end):
     # Select corresponding intervals from attemptsInfoDF
    correspondingAttempts = attemptsInfoDF[(attemptsInfoDF['start_time']>=chunk_start) & (attemptsInfoDF['start_time']<chunk_end)]

    uniqueStarttimes = list(set(correspondingAttempts.loc[:,'start_time']))

    # Create attemptsList
    attemptsList = []
    for startTime in uniqueStarttimes:
        oneAttemptDF = correspondingAttempts[correspondingAttempts['start_time'] == startTime]

        attemptsList.append({
            'attempt_text': getAttemptInfo(oneAttemptDF, 'attempts', 'text'),
            'phones': getAttemptInfo(oneAttemptDF, 'attemptsPhones', 'text'),
            'correct': getAttemptInfo(oneAttemptDF, 'correct', 'text'),
            'description': getAttemptInfo(oneAttemptDF, 'description', 'text'),
            'attempt_start': startTime,
            'attempt_end': getAttemptInfo(oneAttemptDF, 'attempts', 'end_time'),
        })

    return attemptsList

def getPromptDF(basename):

    pathToPromptIdxs = '/vol/tensusers2/wharmsen/SERDA-data/prompts/'

    task = basename.split('-')[1]
    taskType = task.split('_')[0]
    taskNr = task.split('_')[1]

    promptFileName = task + '-wordIDX.csv'
    promptFile = os.path.join(pathToPromptIdxs, promptFileName)

    promptDF = pd.read_csv(promptFile)

    return promptDF

def getPromptID(promptDF, chunk_text, basename):
    # Get index of first row
    firstRowIndex = promptDF.index[0]

    # get index of target word in promptlist
    try:
        target_index = promptDF[promptDF['prompt']==chunk_text].index[0]
    except:
        target_index = -1

    if target_index == firstRowIndex:
        return pd.DataFrame(), promptDF.loc[target_index], promptDF.loc[target_index+1:]
    elif target_index > firstRowIndex:
        return promptDF.loc[firstRowIndex:target_index-1], promptDF.loc[target_index], promptDF.loc[target_index+1:]
    else:
        print('Something went wrong in ' + basename + ': \'', chunk_text, '\' not in promptDF after row ', firstRowIndex)

def alignChunksWithPrompts(promptDF, chunks_df, printable, attemptsInfoDF, basename):

    # Initialize variables
    chunk_dict = {}

    for idx, row in chunks_df.iterrows():

        # Chunk - info
        chunk_start = row['start_time']
        chunk_end = row['end_time']
        chunk_text = row['text']

        if(printable):
            print(chunk_text)

        # List with for each attempt information
        attemptsList = getCorrespondingAttempts(attemptsInfoDF, chunk_start, chunk_end)

        # Get prompt_id that corresponds to chunk_text (idx_last_found_chunk is used to differentiate between multiple occurings of the same word)
        empty_df_rows, target_df_row, remaining_df_rows = getPromptID(promptDF, chunk_text, basename)

        # Create empty dict values for prompts that do not occur in the chunksDF
        for idx, row in empty_df_rows.iterrows():
            # Add empty cases for missing prompts in the chunks
            prompt_id = row['prompt_id']
            chunk_dict[prompt_id] = {
                'chunk_text': '',
                'chunk_start': '',
                'chunk_end': '',
                'attempts': [],
            }

            if(printable):
                print(prompt_id, '-')

        # Create filled dict values for prompts that do occur in the chunksDF
        # Add filled case
        prompt_id = target_df_row['prompt_id']
        chunk_dict[prompt_id] = {
            'chunk_text': chunk_text,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'attempts': attemptsList,
        }
        if(printable):
            print(prompt_id, chunk_text)

        # Update promptDF
        promptDF = remaining_df_rows

    return chunk_dict

def printChunkDict(chunkDict, outputDir, outputName):

    with open(os.path.join(outputDir, outputName), 'w') as f:
        f.write(json.dumps(chunkDict, indent=4, sort_keys=True))


def analyseOneFile(tg_file, outputDir, printable):
    basename = os.path.basename(tg_file).replace('_checked.TextGrid', '')
    print('File that is analyzed:', basename)

    tg_df = read_textgrid_to_dataframe(tg_file)

    chunks_df = initialize_chunks_df(tg_df).reset_index(drop=True)

    attemptsInfoDF = tg_df[tg_df['tier_name'].isin(['attempts', 'attemptsPhones','correct', 'description'])]

    promptDF = getPromptDF(basename)

    chunkDict = alignChunksWithPrompts(promptDF, chunks_df, printable, attemptsInfoDF, basename)

    printChunkDict(chunkDict, outputDir, basename + '.json')

    print('\nLength of chunkDict:', len(chunkDict.keys()))
    print('Length of promptDF:', len(promptDF))
    print('Are lengths equal?', len(chunkDict.keys()) == len(promptDF),'\n')


def run(args):

    # analysis type
    analysisType = args.analysis_type

    # Read output directory and create it if it doesn't exist yet.
    outputDir = args.output_dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Read printable
    printable = False if args.printable in ['False', 'false'] else True

    if analysisType == 'file':

        # Read TextGrid file
        tg_file = args.input_tg_file

        # Analyse Textgrid file
        analyseOneFile(tg_file, outputDir, printable)

    elif analysisType == 'dir':

        # Read TextGrid directory
        tg_dir = args.input_tg_dir

        tg_files = glob.glob(os.path.join(tg_dir, '*.TextGrid'))

        for tg_file in tg_files:
            # Analyse Textgrid file
            analyseOneFile(tg_file, outputDir, printable)

    else:
        print('Set a correct analysisType variable, ', analysisType, 'is incorrect.')
        


    



def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--analysis_type", type=str, help = "Either 'file' or 'dir', depends on whether file or dir of files is analyzed.")
    parser.add_argument("--input_tg_file", type=str, help = "One TextGrid file annotated according to the ASTLA protocol.")
    parser.add_argument("--input_tg_dir", type=str, help = "One directory with TextGrid files annotated according to the ASTLA protocol.")
    parser.add_argument("--output_dir", type=str, help = "Output directory")
    parser.add_argument("--printable", type=str, help="This is a boolean, if True information is printed about where alignment of promptID and chunks goes wrong. This can be used to improve the chunk layer in the textgrid.")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()