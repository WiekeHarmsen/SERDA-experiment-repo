import argparse
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from evaluate import load #https://pypi.org/project/evaluate/ & https://huggingface.co/spaces/evaluate-metric/wer
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score


def read_manual_accuracy_scores(manual_accuracy_dir, task):
    task=task.replace('stories', 'story')
    manualStory1File = os.path.join(manual_accuracy_dir, task+'1_manual_accuracy.csv')
    manualStory2File = os.path.join(manual_accuracy_dir, task+'2_manual_accuracy.csv')
    manualStory3File = os.path.join(manual_accuracy_dir, task+'3_manual_accuracy.csv')

    # Read input files as dataframe
    manualStory1DF = pd.read_csv(manualStory1File, index_col=0)
    manualStory2DF = pd.read_csv(manualStory2File, index_col=0)
    manualStory3DF = pd.read_csv(manualStory3File, index_col=0)

    # Remove speakers with missing accuracy scores
    manualStory1DF = manualStory1DF.dropna()
    manualStory2DF = manualStory2DF.dropna()
    manualStory3DF = manualStory3DF.dropna()

    return manualStory1DF, manualStory2DF, manualStory3DF

def read_automatic_accuracy_scores(asr_accuracy_dir, task):
    task=task.replace('stories', 'story')
    asrStory1File = os.path.join(asr_accuracy_dir, task+'1AsrAccuracyDF.tsv')
    asrStory2File = os.path.join(asr_accuracy_dir, task+'2AsrAccuracyDF.tsv')
    asrStory3File = os.path.join(asr_accuracy_dir, task+'3AsrAccuracyDF.tsv')

    asrStory1DF = pd.read_csv(asrStory1File, index_col=0, sep = '\t')
    asrStory2DF = pd.read_csv(asrStory2File, index_col=0, sep = '\t')
    asrStory3DF = pd.read_csv(asrStory3File, index_col=0, sep = '\t')

    asrStory1DF = asrStory1DF.dropna()
    asrStory2DF = asrStory2DF.dropna()
    asrStory3DF = asrStory3DF.dropna()

    return asrStory1DF, asrStory2DF, asrStory3DF

def swap_correct_incorrect(binary_array):
    binary_array = [9 if x == 0 else 1 for x in binary_array]
    binary_array = [0 if x == 1 else 1 for x in binary_array ]
    return binary_array

def computeEvaluationMetrics(title, y_true, y_pred):

    # By swapping correct and incorrect, we annotate all incorrect words as ones, and all correct words as zeroes.
    y_true = swap_correct_incorrect(y_true)
    y_pred = swap_correct_incorrect(y_pred)
    
    size = "N=" + str(len(y_true))
    acc = "Acc=" + str(round(accuracy_score(y_true, y_pred),3))
    prec = "Prec=" + str(round(precision_score(y_true, y_pred, zero_division=np.nan),3))
    recall = "Recall=" + str(round(recall_score(y_true, y_pred),3))
    f1 = "F1=" + str(round(f1_score(y_true, y_pred, zero_division=np.nan),3))
    mcc = "MCC=" + str(round(matthews_corrcoef(y_true, y_pred),3))

    try:
        auc = "AUC=" + str(round(roc_auc_score(y_true, y_pred),3))
    except:
        auc = "AUC=" + str(np.nan)

    return "    ".join([title, size, acc, prec, recall, f1, mcc, auc])

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def getEvaluationMetrics(manualStoryDF, asrStoryDF):
    # Select only asr accuracy scores that also have manual scoring
    print('#speakers manual:', len(manualStoryDF))
    print('#speakers automatic:', len(asrStoryDF))
    speakersPresentInBothFiles = intersection(list(manualStoryDF.index), list(asrStoryDF.index))
    print('#speakers both:', len(speakersPresentInBothFiles))

    asrStoryDFSelection = asrStoryDF.loc[list(speakersPresentInBothFiles),:]
    manualStoryDFSelection = manualStoryDF.loc[list(speakersPresentInBothFiles),:]

    evalMetricList = []
    for (name, e1, e2) in zip(list(manualStoryDFSelection.index), manualStoryDFSelection.to_numpy(), asrStoryDFSelection.to_numpy()):
        evalMetricList.append(computeEvaluationMetrics(name, e1, e2))

    return evalMetricList, manualStoryDFSelection.to_numpy().flatten(), asrStoryDFSelection.to_numpy().flatten()


def run(args):
    task = args.task
    manual_accuracy_dir = args.manual_accuracy_dir
    manualStory1DF, manualStory2DF, manualStory3DF = read_manual_accuracy_scores(manual_accuracy_dir, task)
    outputDir = args.output_dir
    asrModelName=args.asr_model

    asr_accuracy_dir = args.asr_accuracy_dir
    asrStory1DF, asrStory2DF, asrStory3DF = read_automatic_accuracy_scores(asr_accuracy_dir, task)

    print(len(manualStory1DF.index))
    print(len(asrStory1DF.index))
    print(len(set(manualStory1DF+asrStory1DF)))
    print(sorted(manualStory1DF.index)[0:20])
    print(sorted(asrStory1DF.index)[0:20])

    print('--- STORY 1 ---')
    performanceStringStory1, flatten_scores_manual_story1, flatten_scores_asr_story1 = getEvaluationMetrics(manualStory1DF, asrStory1DF)
    print('\n'.join(performanceStringStory1))

    print('--- STORY 2 ---')
    performanceStringStory2, flatten_scores_manual_story2, flatten_scores_asr_story2 = getEvaluationMetrics(manualStory2DF, asrStory2DF)
    print('\n'.join(performanceStringStory2))

    print('--- STORY 3 ---')
    performanceStringStory3, flatten_scores_manual_story3, flatten_scores_asr_story3 = getEvaluationMetrics(manualStory3DF, asrStory3DF)
    print('\n'.join(performanceStringStory3))

    print('--- OVERALL ---')
    print([len(list(x)) for x in [flatten_scores_manual_story1, flatten_scores_manual_story2, flatten_scores_manual_story3]])
    print([len(list(x)) for x in [flatten_scores_asr_story1, flatten_scores_asr_story2, flatten_scores_asr_story3]])
    all_manual_scores = np.concatenate([flatten_scores_manual_story1, flatten_scores_manual_story2, flatten_scores_manual_story3])
    all_asr_scores = np.concatenate([flatten_scores_asr_story1, flatten_scores_asr_story2, flatten_scores_asr_story3])
    overall_eval_metrics = computeEvaluationMetrics('Total', all_manual_scores, all_asr_scores)    
    print(overall_eval_metrics)
    print(confusion_matrix(all_manual_scores, all_asr_scores))

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    outputFile = os.path.join(outputDir, 'performance_metrics.txt')
    with open(outputFile, 'a') as f:
        f.write('\n\n### DIAGNOSTIC MEASURES '+ str(datetime.now()))

        f.write('\n\nasr model:\t'+ asrModelName)
        f.write('\nmanual accuracy dir:\t'+ manual_accuracy_dir)
        f.write('\nasr accuracy dir:\t'+ asr_accuracy_dir)

        # Output
        f.write('\n--- STORY 1 ---')
        f.write('\n'+'\n'.join(performanceStringStory1))

        f.write('\n--- STORY 2 ---')
        f.write('\n'+'\n'.join(performanceStringStory3))

        f.write('\n--- STORY 3 ---')
        f.write('\n'+'\n'.join(performanceStringStory3))

        f.write('\n--- OVERALL ---')
        f.write('\n'+ overall_eval_metrics)



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