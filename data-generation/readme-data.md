This dataset is generating using SparseLibriMix for overlapping utterances and noise, convolved with CIPIC HRTFs to make the sound binaural.

In order to generate the dataset you need:

- Generate SparseLibriMix using:
    - SparseLibriMix repository: https://github.com/popcornell/SparseLibriMix/tree/master
        - Use overlap = 1, you can adjust this later in generate_binaural.py.
    - LibriSpeech Clean Test Set: https://www.openslr.org/12
    - WHAM!: http://wham.whisper.ai/

    Remember to update the SparseLibriMix script to the correct paths of the resources above.

- CIPIC HRTF Database: https://www.ece.ucdavis.edu/cipic/wp-content/uploads/sites/12/2015/04/cipic_WASSAP_2001_143.pdf

- Update the paths in generate_binaural.py