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

asrFile=$basePath/asr/stories/serda-whispert-story-prompts/2RRDV-story_1-20230113142045040.json
audioFile=$basePath/audio/stories/2RRDV-story_1-20230113142045040.wav
promptFile=/vol/tensusers2/wharmsen/SERDA-data/prompts/story_1.prompt

# asrFile=$basePath/asr/words/.serda-whispert-prompts_unfiltered/2RRDV-words_1_102-20230113140713310.json
# audioFile=$basePath/audio/words/segments/2RRDV-words_1_102-20230113140713310.wav
# promptFile=$basePath/prompts/words/2RRDV-words_1_102-20230113140713310.prompt

# python3 ./02-stories-align-prompt-whispert.py --analysis_type 'file' --output_dir $basePath/test/ --input_audio $audioFile --input_asr_result $asrFile --input_prompt $promptFile

python3 ./03-tg2csv.py --input_tg '/vol/tensusers2/wharmsen/SERDA-annotations/round1_stories_all_marjul/textgrid/set1_jul/ZPGND-story_3-20230116114853113_checked.TextGrid' --output_dir $basePath/test/

