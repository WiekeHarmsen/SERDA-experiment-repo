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

audioFile='/vol/tensusers2/wharmsen/SERDA/round1/audio/stories/5NMJH-story_3-20221107131303195.wav'
asr_result='/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/stories/whispert_dis/json-asr-results/5NMJH-story_3-20221107131303195.json'
prompt='/vol/tensusers2/wharmsen/SERDA-data/prompts/story_3.prompt'
output_dir_csv_alignments='/vol/tensusers5/wharmsen/SERDA-experiment-repo/12_fluency_whisperT'

python3 ./01-stories-align-prompt-whispert.py --analysis_type 'file' --input_audio $audioFile --input_asr_result $asr_result --input_prompt $prompt --output_dir $output_dir_csv_alignments
