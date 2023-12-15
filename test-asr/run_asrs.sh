#!/bin/bash

audio_words_segments_round1=/vol/tensusers2/wharmsen/SERDA/round1/audio/words/segments/

python3 wav2vec2-large-xlsr-53-dutch.py --audio_dir $audio_words_segments_round1 --asr_model 'wav2vec2-large-xlsr-53-dutch' --align_model 'adagt' --round 'round1' --task_type 'words' --output_dir '/vol/tensusers2/wharmsen/SERDA-experiment-data'