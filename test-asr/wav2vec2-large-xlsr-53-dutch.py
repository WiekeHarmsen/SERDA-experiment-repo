"""
Dependent on '/vol/tensusers2/wharmsen/SERDA-data/prompts/round1_speakerIDs.csv'

"""

# import torch
# import librosa
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import pandas as pd
import os
import glob
import numpy as np

import argparse
import alignment_adagt as adagt
import asr_decoders

"""
Get unique wordIDs
"""
def getWordIDs(task):
    if(task == 'words_1'):
        return np.arange(101, 151, 1)
    elif(task == 'words_2'):
        return np.arange(201, 251, 1)
    elif(task == 'words_3'):
        return np.arange(301, 351, 1)
    else:
        print("The task is unknown:", task)

"""
Get unique speakerIDs

round:  either 'round1' or 'round2'
"""
def getSpeakerIDs(round):
    df =  pd.read_csv('/vol/tensusers2/wharmsen/SERDA-data/prompts/' + round + '_speakerIDs.csv')
    return list(df['round1_speaker_ids'])


# def decode(audio_file):

#     LANG_ID = "nl"
#     MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
#     SAMPLES = 10

#     processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
#     model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

#     audio, sr = librosa.load(audio_file, sr=16_000)

#     inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     predicted_sentences = processor.batch_decode(predicted_ids)

#     return predicted_sentences

def getPrompt(task, prompt_id):

    # Create path to prompt file
    pathToPromptIdxs = '/vol/tensusers2/wharmsen/SERDA-data/prompts/'
    promptFileName = task + '-wordIDX.csv'
    promptFile = os.path.join(pathToPromptIdxs, promptFileName)

    # Read prompt file
    promptDF = pd.read_csv(promptFile)

    # Reset index
    promptDF = promptDF.set_index('prompt_id')
    
    # Return prompt corresponding to promptID
    return promptDF.loc[prompt_id,'prompt']

def run(args):

    round = args.round
    taskType = args.task_type
    outputDir = args.output_dir
    asrModel = args.asr_model
    alignModel = args.align_model
    audioDir = args.audio_dir

    score_df_output = 'words_x-AsrAccuracy-w2v-m1.csv'

    outputDirSpecified = os.path.join(outputDir, asrModel)
    if not os.path.exists(outputDirSpecified):
        os.makedirs(outputDirSpecified)

    if(taskType == 'words'):

        wordAccDict = {}

        uniqueStudents = getSpeakerIDs(round)

        # Get unique student IDs
        for task in ['words_1']: #['words_1', 'words_2', 'words_3']:
            # Initialize alignmentDF
            alignmentList = []

            # Get word IDs for the word task
            word_ids = getWordIDs(task)

            # Initialize outputDFs, one for each task
            words_asr_acc_df = pd.DataFrame(index=uniqueStudents, columns=word_ids)

            # Fill DF
            for speaker_id in words_asr_acc_df.index[0:5]:
                for prompt_id in words_asr_acc_df.columns[0:3]:

                    # get audio segment
                    audio_file = speaker_id + '-' + task + '_' + str(prompt_id) + '*.wav'
                    audioPath = glob.glob(os.path.join(audioDir, audio_file))

                    # decode segment using selected asr model
                    asr_result = asr_decoders.getAsrResult(asrModel, [audioPath[0]])
                    asr_trans = asr_result[0]['transcription']
                    print(asr_trans)

                    # get prompt transcription
                    prompt = getPrompt(task, prompt_id)

                    # align asr transcription and prompt and get correctness value
                    oneWordAlignmentDF = adagt.two_way_alignment(prompt, asr_trans)
                    alignmentList.append([oneWordAlignmentDF.index[0], oneWordAlignmentDF.iloc[0,0], oneWordAlignmentDF.iloc[0,1], oneWordAlignmentDF.iloc[0,2]])

                    print(alignmentList[-1])
                    
                    words_asr_acc_df.loc[speaker_id, prompt_id] = 1 if oneWordAlignmentDF.iloc[0,2] == True else 0
                
                align_df_output = 'words_x-AsrAccuracy-w2v-m1-align.csv'
                pd.DataFrame(alignmentList, columns = ['prompt', 'align', 'rev_align', 'correct']).to_csv(os.path.join(outputDirSpecified, align_df_output.replace('words_x', task)))


            # Print DF
            words_asr_acc_df.to_csv(os.path.join(outputDirSpecified, score_df_output.replace('words_x', task)))


    else:
        print('taskType == story is not yet implemented')


def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--audio_dir", type=str, help = "Audio directory")
    parser.add_argument("--round", type=str, help = "round1 or round2, used to get the studentIDs")
    parser.add_argument("--task_type", type=str, help = "words or story")
    parser.add_argument("--output_dir", type=str, help = "Output directory to save STUDENT x PROMPT_ID csv files with ASR generated accuracy scores.")
    parser.add_argument("--asr_model", type=str, help = "Name of model used for asr decoding")   
    parser.add_argument("--align_model", type=str, help = "Name of model used for alignment")   
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

