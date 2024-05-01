import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_manual_scores(manual_accuracy_dir, task, score_type):
    task=task.replace('stories', 'story')

    manualStory1File = os.path.join(manual_accuracy_dir, task+'1_manual_' + score_type + '.csv')
    manualStory2File = os.path.join(manual_accuracy_dir, task+'2_manual_' + score_type + '.csv')
    manualStory3File = os.path.join(manual_accuracy_dir, task+'3_manual_' + score_type + '.csv')

    # Read input files as dataframe
    manualStory1DF = pd.read_csv(manualStory1File, index_col=0)
    manualStory2DF = pd.read_csv(manualStory2File, index_col=0)
    manualStory3DF = pd.read_csv(manualStory3File, index_col=0)

    # print('a', [len(x) for x in [manualStory1DF, manualStory2DF,manualStory3DF]])

    # Remove speakers with missing accuracy scores
    manualStory1DF = manualStory1DF.dropna()
    manualStory2DF = manualStory2DF.dropna()
    manualStory3DF = manualStory3DF.dropna()

    # print('b', [len(x) for x in [manualStory1DF, manualStory2DF,manualStory3DF]])


    return manualStory1DF, manualStory2DF, manualStory3DF

def read_automatic_scores(asr_accuracy_dir, task, score_type):

    task=task.replace('stories', 'story')

    asrStory1File = os.path.join(asr_accuracy_dir, task+'1Asr' + score_type + 'DF.tsv')
    asrStory2File = os.path.join(asr_accuracy_dir, task+'2Asr' + score_type + 'DF.tsv')
    asrStory3File = os.path.join(asr_accuracy_dir, task+'3Asr' + score_type + 'DF.tsv')

    asrStory1DF = pd.read_csv(asrStory1File, index_col=0, sep = '\t')
    asrStory2DF = pd.read_csv(asrStory2File, index_col=0, sep = '\t')
    asrStory3DF = pd.read_csv(asrStory3File, index_col=0, sep = '\t')

    # print('1', [len(x) for x in [asrStory1DF, asrStory2DF,asrStory3DF]])

    asrStory1DF = asrStory1DF.dropna()
    asrStory2DF = asrStory2DF.dropna()
    asrStory3DF = asrStory3DF.dropna()

    # print('2', [len(x) for x in [asrStory1DF, asrStory2DF,asrStory3DF]])

    return asrStory1DF, asrStory2DF, asrStory3DF

def extractStats(differenceStoryDF, taskName):
    outputDescriptiveStatsOneTask = []
    for studentID, row in differenceStoryDF.iterrows():
            
        scores = np.array(row).flatten()

        descriptiveStats = pd.Series(scores).describe()
        count = descriptiveStats['count']
        mean = descriptiveStats['mean']
        std = descriptiveStats['std']
        minn = descriptiveStats['min']
        perc25 = descriptiveStats['25%']
        perc50 = descriptiveStats['50%']
        perc75 = descriptiveStats['75%']
        maxx = descriptiveStats['max']

        # The mean should be very close to zero
        outputDescriptiveStatsOneTask.append('\t'.join([taskName, studentID]+[str(round(nr, 3)) for nr in [mean, std, minn, perc25, perc50, perc75, maxx]]))
    return np.array(outputDescriptiveStatsOneTask)

def run(args):
    manual_asr_score_dict = {  'chunk_starttime': 'StartSpeak', 
                        'finalattempt_starttime': 'StartSpeak',
                        'chunk_endtime': 'StopSpeak',                        
                        'finalattempt_endtime': 'StopSpeak',
                        'chunk_duration': 'Time',
                        'finalattempt_duration': 'Time'
                        }
    
    task = args.task
    manual_accuracy_dir = args.manual_accuracy_dir
    outputDir = args.output_dir
    asrModelName=args.asr_model

    asr_accuracy_dir = '/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories_manann_11jan/whispert_dis/csv-scores'
    header = ['\t'.join(["task", "studentID", "mean", "std", "min", "perc25", "perc50", "perc75", "max"])]

    # print(['story1', 'story2', 'story3'])
    outputFile = os.path.join(outputDir, 'performance_metrics.txt')
    with open(outputFile, 'a') as f:
        f.write('\n\n### TIMING MEASURES ')
        f.write('\n' + "manual_variable" + '\t' + 'asr_variable'+ '\t\t\t' + 'mean_of_file_means')

    outputDescriptiveStats = []
    for (manual_key, asr_key) in manual_asr_score_dict.items():

        # Read the scores
        manualStory1DF, manualStory2DF, manualStory3DF = read_manual_scores(manual_accuracy_dir, task, manual_key)

        # Preprocess ASR DFs
        manualStory1DF = manualStory1DF.replace(0.0, np.nan).replace(999, np.nan)
        manualStory2DF = manualStory2DF.replace(0.0, np.nan).replace(999, np.nan)
        manualStory3DF = manualStory3DF.replace(0.0, np.nan).replace(999, np.nan)

        # print('c', [len(x) for x in [manualStory1DF, manualStory2DF,manualStory3DF]])

        # Preprocess Manual DFs
        asrStory1DF, asrStory2DF, asrStory3DF = read_automatic_scores(asr_accuracy_dir, task, asr_key)

        asrStory1DF = asrStory1DF.loc[manualStory1DF.index].replace(0.0, np.nan).replace(999, np.nan)
        asrStory2DF = asrStory2DF.loc[manualStory2DF.index].replace(0.0, np.nan).replace(999, np.nan)
        asrStory3DF = asrStory3DF.loc[manualStory3DF.index].replace(0.0, np.nan).replace(999, np.nan)

        # print('3', [len(x) for x in [asrStory1DF, asrStory2DF,asrStory3DF]])

        # Subtract DFs
        differenceStory1DF = manualStory1DF - asrStory1DF
        differenceStory2DF = manualStory2DF - asrStory2DF
        differenceStory3DF = manualStory3DF - asrStory3DF

        # print('DIFF', [len(x) for x in [differenceStory1DF, differenceStory2DF,differenceStory3DF]])

        # Compute statistics for each speaker
        statsStory1 = extractStats(differenceStory1DF, 'story1')
        statsStory2 = extractStats(differenceStory2DF, 'story2')
        statsStory3 = extractStats(differenceStory3DF, 'story3')

        outputDescriptiveStats = list(np.concatenate((header, statsStory1, statsStory2, statsStory3), axis=0))
        
        outputDirFolder = os.path.join(outputDir, 'timing_measures')
        if not os.path.exists(outputDirFolder):
            os.makedirs(outputDirFolder)

        with open(os.path.join(outputDirFolder, manual_key + '-' + asr_key + '-measures.tsv'), 'w') as f:
            f.write('\n'.join(outputDescriptiveStats))

        data = [line.split('\t') for line in outputDescriptiveStats]
        meanValue = pd.DataFrame(data[1:], columns=data[0])['mean'].astype(float).mean()

        with open(outputFile, 'a') as f:
            f.write('\n' + manual_key + '\t' + asr_key+ '\t\t\t' + str(round(meanValue,3)))

    with open(outputFile, 'a') as f:
        f.write('\nFor detailed measures, see folder timing_measures')


def main():
    parser = argparse.ArgumentParser("Message")
    parser.add_argument("--asr_model", type=str, help = "The name of the ASR model.")
    parser.add_argument("--task", type=str, help = "either story or words")
    parser.add_argument("--manual_accuracy_dir", type=str, help = "studentID x prompt accuracy file")
    parser.add_argument("--asr_accuracy_dir", type=str, help = "studentID x prompt accuracy file")

    parser.add_argument("--output_dir", type=str, help = "Output directory - evaluation measures")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()