import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_manual_scores(manual_accuracy_dir, task, score_type):
    manualStory1File = os.path.join(manual_accuracy_dir, task+'1_manual_' + score_type + '.csv')
    manualStory2File = os.path.join(manual_accuracy_dir, task+'2_manual_' + score_type + '.csv')
    manualStory3File = os.path.join(manual_accuracy_dir, task+'3_manual_' + score_type + '.csv')

    # Read input files as dataframe
    manualStory1DF = pd.read_csv(manualStory1File, index_col=0)
    manualStory2DF = pd.read_csv(manualStory2File, index_col=0)
    manualStory3DF = pd.read_csv(manualStory3File, index_col=0)

    # Remove speakers with missing accuracy scores
    manualStory1DF = manualStory1DF.dropna()
    manualStory2DF = manualStory2DF.dropna()
    manualStory3DF = manualStory3DF.dropna()

    return manualStory1DF, manualStory2DF, manualStory3DF

def read_automatic_scores(asr_accuracy_dir, task, score_type):

    asrStory1File = os.path.join(asr_accuracy_dir, task+'1Asr' + score_type + 'DF.tsv')
    asrStory2File = os.path.join(asr_accuracy_dir, task+'2Asr' + score_type + 'DF.tsv')
    asrStory3File = os.path.join(asr_accuracy_dir, task+'3Asr' + score_type + 'DF.tsv')

    asrStory1DF = pd.read_csv(asrStory1File, index_col=0, sep = '\t')
    asrStory2DF = pd.read_csv(asrStory2File, index_col=0, sep = '\t')
    asrStory3DF = pd.read_csv(asrStory3File, index_col=0, sep = '\t')

    asrStory1DF = asrStory1DF.dropna()
    asrStory2DF = asrStory2DF.dropna()
    asrStory3DF = asrStory3DF.dropna()

    return asrStory1DF, asrStory2DF, asrStory3DF

# python3 ./eval-stories-diag-metrics.py --asr_model $modelname --task $task --manual_accuracy_dir $csv_accuracy_scores_manual --asr_accuracy_dir $csv_accuracy_scores_automatic --output_dir $output_dir_eval_metrics
# def main():
#     parser = argparse.ArgumentParser("Message")
#     parser.add_argument("--asr_model", type=str, help = "The name of the ASR model.")
#     parser.add_argument("--task", type=str, help = "either story or words")
#     parser.add_argument("--manual_accuracy_dir", type=str, help = "studentID x prompt accuracy file")
#     parser.add_argument("--asr_accuracy_dir", type=str, help = "studentID x prompt accuracy file")

#     parser.add_argument("--output_dir", type=str, help = "Output directory - evaluation measures")
#     parser.set_defaults(func=run)
#     args = parser.parse_args()
#     args.func(args)

# if __name__ == "__main__":
#     main()