# Audio fingerprinting system

An audio fingerprinting and recognition system that can identify songs from short audio samples.

# Overview:

This project implements an audio fingerprinting algorithm similar to Shazam's approach. It creates unique fingerprints from audio files by:

Computing the spectrogram of audio signals
Finding spectral peaks to create a constellation map
Creating anchor points and target zones
Generating hash values from pairs of anchor and target points
Matching query audio against a database of pre-computed fingerprints

# Features

Create fingerprint database from a collection of audio files
Identify songs from short audio clips (snippets)
Support for different audio genres (classical and pop)
Performance evaluation and accuracy reporting
Audio pre-processing capabilities 

# Dependencies

Python 3.6+
NumPy
Librosa
SciPy
Scikit-image
Matplotlib
NoiseReduce (optional for noise reduction)

Install dependencies:
pip install numpy librosa scipy scikit-image matplotlib noisereduce

# Usage
The below 2 functions can be called from main.py: 

fingerprintBuilder(/path/to/database/,/path/to/fingerprints/)
audioIdentification(/path/to/queryset/,/path/to/fingerprints/,/path/to/output.txt)
The format of the output.txt file will include one line for each query audio recording, using the
following format:
query audio 1.wav database audio x1.wav database audio x2.wav database audio x3.wav
query audio 2.wav database audio y1.wav database audio y2.wav database audio y3.wav
...
where the filenames in each line are separated by a tab, and the three database recordings per line are
the top three ones which are identified by the audio identification system as more closely matching
the query recording, ranked from first to third.

Parameter Tuning
The system's performance can be optimized by adjusting several parameters:

min_distance: Minimum distance between peaks in the spectrogram
hop_size: Hop length for STFT computation
cutoff: Cutoff frequency for high-pass filtering
threshold: Threshold for peak detection
sr_downsample: Sample rate for downsampling audio files
target_zone_time: Time range for target zone in time bins
target_zone_freq: Frequency range for target zone in frequency bins
fan_out: Maximum number of target points per anchor

# Performance Evaluation
The system measures recognition accuracy in terms of:

Overall accuracy (percentage of correctly identified samples)
Genre-specific accuracy (classical vs. pop)
Top-1 and Top-3 match accuracy

Results are saved to result.txt for each run. The current file has results for a proportion of runs on the whole dataset, and rest of them on a few others.

notes_2 contains a list of songs that are same but labelled differently.
How It Works
Fingerprinting Process

Pre-processing: Load audio and optionally apply filtering or noise reduction
Spectrogram Computation: Convert audio to time-frequency representation
Peak Detection: Find local maxima in the spectrogram
Anchor Points: Select certain peaks as anchors
Target Zones: Define zones around each anchor to find target points
Hash Generation: Create hashes from pairs of anchor and target points
Database Storage: Save all hashes with reference to their source audio

# Matching Process

Query Processing: Generate fingerprint hashes for the query audio
Hash Matching: Compare query hashes with database hashes
Score Calculation: Count matching hashes for each song in the database
Ranking: Return the songs with the highest match scores

Customization
Audio Pre-processing
You can enable additional pre-processing steps by uncommenting these lines in search_hash:
python
# reduced_noise = nr.reduce_noise(y=y, sr=sr)
# y = highpass_filter_audio(y, sr, cutoff_freq=cutoff)
Algorithm Parameters
Adjust the global parameters at the top of main.py to optimize for your specific audio dataset:
pytho
nmin_distance = 6       # Adjust for peak density
hop_size = 128         # Lower values for higher time resolution
cutoff = 100           # Increase to remove more low-frequency noise
threshold = 0.001      # Lower values detect more peaks
sr_downsample = 11025  # Sample rate for processing
References
This implementation is based on the audio fingerprinting concepts described in:

Wang, A. (2003). "An Industrial-Strength Audio Search Algorithm." In ISMIR.
Cano, P., et al. (2005). "A Review of Audio Fingerprinting." Journal of VLSI Signal Processing.

To know more about the underlying concepts and development process visit my blog here: https://medium.com/@shivam01110011/audio-fingerprinting-aee18fb88d4a

License
MIT License
