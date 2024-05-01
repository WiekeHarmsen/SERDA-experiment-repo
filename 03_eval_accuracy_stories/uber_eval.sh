#!/bin/bash

# Generic inputs

modelname=whispert_dis #Choose the right 01 script
round=round1 #Choose the right 01 script
task=stories #either story or words
subset=stories_manann_11jan

# Input WER script
output_dir_csv_alignments=/vol/tensusers2/wharmsen/SERDA-experiment-data/$round/$subset/$modelname/csv-alignments
output_dir_eval_metrics=/vol/tensusers2/wharmsen/SERDA-experiment-data/$round/$subset/$modelname/eval-metrics

# Input diagnostic measures script
csv_accuracy_scores_automatic=/vol/tensusers2/wharmsen/SERDA-experiment-data/$round/$subset/$modelname/csv-scores

# Manual transcriptions
manual_transcription_tsv=/vol/tensusers2/wharmsen/SERDA-annotations/round1_stories_all_11jan/02_json/asr-transcriptions.tsv
csv_accuracy_scores_manual=/vol/tensusers2/wharmsen/SERDA-annotations/round1_stories_all_11jan/03_scoring_csv
# csv_accuracy_scores_manual=/vol/tensusers2/wharmsen/SERDA-annotations/round2a_words_bra

# Compute WER and CER
# python3 ./eval-stories-WER.py --asr_model $modelname --manual_trans_file $manual_transcription_tsv --asr_trans_file $output_dir_csv_alignments/asr-transcriptions.tsv --output_dir $output_dir_eval_metrics

# Compute diagnostic measures
# python3 ./eval-stories-diag-metrics.py --asr_model $modelname --task $task --manual_accuracy_dir $csv_accuracy_scores_manual --asr_accuracy_dir $csv_accuracy_scores_automatic --output_dir $output_dir_eval_metrics

# Compute diagnostic measures
python3 ./eval_stories_timing_metrics.py --asr_model $modelname --task $task --manual_accuracy_dir $csv_accuracy_scores_manual --asr_accuracy_dir $csv_accuracy_scores_automatic --output_dir $output_dir_eval_metrics
