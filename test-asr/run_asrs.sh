#!/bin/bash

audio_words_segments_round1=/vol/tensusers2/wharmsen/SERDA/round1/audio/words/segments/

# python3 wav2vec2-large-xlsr-53-dutch.py --audio_dir $audio_words_segments_round1 --asr_model 'wav2vec2-large-xlsr-53-dutch' --align_model 'adagt' --round 'round1' --task_type 'words' --output_dir '/vol/tensusers2/wharmsen/SERDA-experiment-data'

python3 ctc-segmentation-kurzinger.py --audio_file "/vol/tensusers2/wharmsen/SERDA/round1/audio/stories/2RRDV-story_1-20230113142045040.wav" --cache_path '/vol/tensusers5/wharmsen/wav2vec2/cache/' --model "fc" --output_dir "/vol/tensusers2/wharmsen/SERDA-experiment-data/round1/words/segments/ctc-kurz"