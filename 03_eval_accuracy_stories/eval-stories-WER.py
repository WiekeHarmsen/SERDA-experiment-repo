"""
This script takes two TSV files as input.
TSV1    ['filename', 'manual_transcript']
TSV2    ['filename', 'asr_transcript']

The output are descriptive statistics and WER/CER scores.
"""

import argparse
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from evaluate import load #https://pypi.org/project/evaluate/ & https://huggingface.co/spaces/evaluate-metric/wer
from datetime import datetime


def run(args):

    # Read inputs
    asrModel = args.asr_model
    manualTransFile = args.manual_trans_file
    asrTransFile = args.asr_trans_file
    outputDir = args.output_dir

    # Create output dir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Import manual transcriptions
    manualTransDF = pd.read_csv(manualTransFile, sep='\t', header=None).drop_duplicates()
    manualTransDF.columns = ['filename', 'manual_transcript']
    manualTransDF = manualTransDF.set_index('filename')

    # Import asr transcriptions
    asrTransDF = pd.read_csv(asrTransFile, sep='\t').drop_duplicates()
    asrTransDF.columns = ['filename', 'asr_transcript']
    asrTransDF = asrTransDF.set_index('filename')

    # Merge ASR and manual transcriptions
    combinedDF = pd.concat([manualTransDF, asrTransDF], axis=1).dropna(axis=0)

    print('# manual transcriptions:',  len(manualTransDF))
    print('# asr transcriptions',  len(asrTransDF))
    print('# combined transcriptions:', len(combinedDF))

    combinedDF['length_words_orttrans'] = combinedDF['manual_transcript'].apply(lambda x: len(x.split(' ')))
    combinedDF['length_chars_orttrans'] = combinedDF['manual_transcript'].apply(lambda x: len(x))

    manual_transcriptions = list(combinedDF['manual_transcript'])
    asr_transcriptions = list(combinedDF['asr_transcript'])

    # WER metric
    wer_metric=load("wer") #load_metric('wer')
    wer = round(wer_metric.compute(predictions=asr_transcriptions, references=manual_transcriptions), 3)

    # CER metric
    cer_metric=load("cer") #load_metric('cer')
    cer = round(cer_metric.compute(predictions=asr_transcriptions, references=manual_transcriptions), 3)

    # Output measures
    output_measures = [asrModel, combinedDF['length_words_orttrans'].mean(), combinedDF['length_chars_orttrans'].mean(), wer, cer]
    output_string = '\t'.join([str(x) for x in output_measures])

    with open(os.path.join(outputDir, 'performance_metrics.txt'), 'w') as f:
        f.write('### EXPERIMENT RESULTS '+ str(datetime.now()))

        f.write('\n\nasr model:\t'+ asrModel)
        f.write('\northographic transcriptions:\t'+ manualTransFile)
        f.write('\nasr transcriptions:\t'+ asrTransFile)

        f.write('\n\n# manual transcriptions:\t'+  str(len(manualTransDF)))
        f.write('\n# asr transcriptions\t'+  str(len(asrTransDF)))
        f.write('\n# combined transcriptions:\t'+ str(len(combinedDF)))

        # Output
        f.write('\n\n' + '\t'.join(['asrModel', 'mean_ort_length_in_words', 'mean_ort_length_in_chars', 'wer', 'cer']))
        f.write('\n' + output_string)
        f.write('\nWER of 0.2 means that 20\% of words in reference are incorrectly recognized by ASR.')
        f.write('\nCER of 0.2 means that 20\% of characters in reference are incorrectly recognized by ASR.')


    print('\t'.join(['asrModel', 'mean_len_words', 'mean_len_chars', 'wer', 'cer']))
    print(output_string)
    print('See results in:', os.path.join(outputDir, 'performance_metrics.txt'))


def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--asr_model", type=str, help = "asr-transcriptions.tsv containing asr transcriptions")
    parser.add_argument("--manual_trans_file", type=str, help = "asr-transcriptions.tsv containing asr transcriptions")
    parser.add_argument("--asr_trans_file", type=str, help = "asr-transcriptions.tsv containing orthographic transcriptions")

    parser.add_argument("--output_dir", type=str, help = "Output directory - evaluation measures")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()