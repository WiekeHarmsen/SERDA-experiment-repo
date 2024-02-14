#!/bin/bash

# Before running this script, do the following steps:
# 1. Activate virtual environment:
# $ source virenv-wav2vec2/bin/activate
# 2. Set basePath variable
# 3. Create folder named "output_serda" in basePath and put urls.txt and all_results.zip (both downloaded from SERDA app) in this folder
# 4. Navigate to the dir where this file is placed
# $ cd ./SERDA-experiments
# 5. Execute this file.
# $ nohup ./uber.sh &


basePath='/vol/tensusers2/wharmsen/SERDA/round1'

###
# Analyse one file
###

# asrFile=$basePath/asr/stories/serda-whispert-story-prompts/2RRDV-story_1-20230113142045040.json
# audioFile=$basePath/audio/stories/2RRDV-story_1-20230113142045040.wav
# promptFile=/vol/tensusers2/wharmsen/SERDA-data/prompts/story_1.prompt

# asrFile=$basePath/asr/words/.serda-whispert-prompts_unfiltered/2RRDV-words_1_102-20230113140713310.json
# audioFile=$basePath/audio/words/segments/2RRDV-words_1_102-20230113140713310.wav
# promptFile=$basePath/prompts/words/2RRDV-words_1_102-20230113140713310.prompt


# python3 ./01-stories-align-prompt-whispert.py --analysis_type 'file' --input_audio $audioFile --input_asr_result $asrFile --input_prompt $promptFile --output_dir $basePath/test/ 
# python3 ./02-stories-convert-csv.py --csv_dir $basePath/test/ --output_dir $basePath/test/test

###
# Analyse a directory
###

# Round1 ManAnn_11jan inputs
# audioDir=/vol/tensusers2/wharmsen/SERDA-data/round1/audio/stories/selection_manann_11jan
# asrDir=/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories_manann_11jan/$modelname/json-asr-results #whispert
# # asrDir=/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories_manann_11jan/$modelname/json_recognized #ctc_kurz_w2v_fc
# output_dir_csv_alignments=/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories_manann_11jan/$modelname/csv-alignments
# output_dir_csv_accuracy=/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories_manann_11jan/$modelname/csv-scores
# output_dir_eval_metrics=/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories_manann_11jan/$modelname/eval-metrics

# Generic inputs
promptDir=/vol/tensusers2/wharmsen/SERDA-data/prompts

modelname=whispert_dis #Choose the right 01 script
round=round2 #Choose the right 01 script
task=stories #stories
subset=stories #stories

audioDir=/vol/tensusers2/wharmsen/SERDA-data/$round/audio/$task
#audioDir=/vol/tensusers2/wharmsen/SERDA/$round/audio/$task/full # This is the audio dir for round1 audio (SERDA ipv SERDA-data)
asrDir=/vol/tensusers2/wharmsen/SERDA-experiment-data/$round/$subset/$modelname/json-asr-results #whispert

output_dir_csv_alignments=/vol/tensusers2/wharmsen/SERDA-experiment-data/$round/$subset/$modelname/csv-alignments
output_dir_csv_accuracy=/vol/tensusers2/wharmsen/SERDA-experiment-data/$round/$subset/$modelname/csv-scores

python3 ./01-stories-align-prompt-whispert.py --analysis_type 'dir' --input_audio_dir $audioDir --input_asr_dir $asrDir --input_prompt_dir $promptDir --output_dir $output_dir_csv_alignments
# python3 ./01-stories-align-prompt-ctc-kurz-w2v.py --analysis_type 'dir' --input_audio_dir $audioDir --input_asr_dir $asrDir --input_prompt_dir $promptDir --output_dir $output_dir_csv_alignments 
python3 ./02-stories-convert-csv.py --csv_dir $output_dir_csv_alignments --task_type $task --round $round --output_dir $output_dir_csv_accuracy
