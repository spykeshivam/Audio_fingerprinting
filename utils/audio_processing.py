import numpy as np
import librosa
import os
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as wavfile
from scipy import signal


def plot_peaks(D,coordinates):
    """
    Plot a spectrogram and constellation map of detected peaks.
    
    Args:
        D (numpy.ndarray): STFT spectrogram matrix
        coordinates (numpy.ndarray): Array of peak coordinates [frequency_bin, time_bin]
    """
    plt.figure(figsize=(10/3, 5))
    librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='cqt_note', x_axis='time',sr=22050)

    # Detect peaks from STFT spectrogram and plot constellation map
    plt.figure(figsize=(10/3, 5))
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    plt.show()

"""
def detect_peaks(song, threshold_rel=0.01, high_freq_boost=1.5):
    #Loads an audio file and detects peaks in its spectrogram with higher frequency emphasis.
    y, sr = librosa.load(song)
    D = np.abs(librosa.stft(y, n_fft=1024, window='hann', win_length=1024, hop_length=256))
    
    # Create frequency weighting (higher weights for higher frequencies)
    freq_bins = D.shape[0]
    weights = np.linspace(1.0, high_freq_boost, freq_bins).reshape(-1, 1)
    
    # Apply weights to spectrogram
    D_weighted = D * weights
    
    # Find peaks in the weighted spectrogram
    coordinates = peak_local_max(np.log(D_weighted + 1e-10), min_distance=15, threshold_rel=threshold_rel)
    
    return coordinates, D.shape, D
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from skimage.feature import peak_local_max

def detect_peaks(y, sr, threshold_rel=0.01, min_distance=10, hop_size=256, files=True):
    """
    Detect spectral peaks in an audio signal's spectrogram and plot two spectrograms: 
    one without and one with peak markers.
    
    Args:
        y (numpy.ndarray): Audio time series
        sr (int): Sample rate
        threshold_rel (float): Relative threshold for peak detection 
        min_distance (int): Minimum distance between peaks 
        hop_size (int): Hop length for STFT
    
    Returns:
        tuple: (coordinates, spectrogram_shape, spectrogram)
    """
    # Compute magnitude spectrogram
    D = np.abs(librosa.stft(y, n_fft=1024, window='hann', win_length=1024, hop_length=hop_size))
    
    # Convert to log scale
    log_D = np.log(D + 1e-10)
    
    # Detect spectral peaks
    coordinates = peak_local_max(log_D, min_distance=min_distance, threshold_rel=threshold_rel)
    if files:
        # Plot 1: Log spectrogram without peaks
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(log_D, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

        # Plot 2: Log spectrogram with detected peaks
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(log_D, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log', cmap='magma')
        plt.scatter(coordinates[:, 1] * hop_size / sr, 
                    librosa.core.fft_frequencies(sr=sr, n_fft=1024)[coordinates[:, 0]], 
                    marker='o', color='cyan', s=10, label='Peaks')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log Spectrogram with Detected Peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return coordinates, D.shape, D



def analyze_and_visualize(coordinates, spectrogram_shape, D, target_zone_time=50, target_zone_freq=80, fan_out=30):
    """
    Create an anchor dictionary with target zones for each anchor point.
    
    This function implements the audio fingerprinting algorithm by:
    1. Sorting peak coordinates by time
    2. For each anchor point, defining a target zone
    3. Finding other peaks within this target zone
    4. Creating pairs of anchor and target points
    
    Args:
        coordinates (numpy.ndarray): Array of peak coordinates [frequency_bin, time_bin]
        spectrogram_shape (tuple): Shape of the STFT spectrogram
        D (numpy.ndarray): STFT spectrogram matrix
        target_zone_time (int): Target zone width in time bins
        target_zone_freq (int): Target zone height in frequency bins
        fan_out (int): Maximum number of target points per anchor
    
    Returns:
        dict: Dictionary with anchor points as keys and lists of target points as values
    """
    # Sort coordinates by time
    sorted_coords = coordinates[coordinates[:, 1].argsort()]
    anchor_dict = {}

    # For each peak, create an anchor point and find targets in its zone
    for i, anchor in enumerate(sorted_coords):
        # Define the target zone boundaries (a rectangular area after the anchor)
        min_time = anchor[1] + 1
        max_time = anchor[1] + target_zone_time
        min_freq = max(0, anchor[0] - target_zone_freq)
        max_freq = min(spectrogram_shape[0], anchor[0] + target_zone_freq)
        # Find all peaks within the target zone
        candidates = [
            point for j, point in enumerate(sorted_coords) if j != i
            and min_time <= point[1] <= max_time and min_freq <= point[0] <= max_freq
        ]
        # Sort candidates by their magnitude in the spectrogram
        candidates_sorted = sorted(candidates, key=lambda p: D[int(p[0]), int(p[1])], reverse=True)
        #print(len(candidates_sorted))
        # Take only the top fan_out number of candidates
        anchor_dict[tuple(anchor)] = candidates_sorted[:fan_out]

    return anchor_dict

def hash_anchors(anchor_dict):
    """
    Generate hashes from anchor points and their target points.
    
    For each anchor-target pair, creates a hash based on their frequency and time delta.
    
    Args:
        anchor_dict (dict): Dictionary with anchor points as keys and lists of target points as values
    
    Returns:
        list: List of hashes, where each hash is [f1, f2, t_delta]
            - f1: Frequency of anchor point
            - f2: Frequency of target point
            - t_delta: Time difference between target and anchor
    """
    hashes = []
    # Process each anchor point and its targets
    for anchor, target_points in anchor_dict.items():
        # Extract frequency and time coordinates of the anchor
        f1, t1 = anchor
        # For each target point paired with this anchor
        for target in target_points:
            # Extract frequency and time coordinates of the target
            f2, t2 = target
            # Compute the time difference between target and anchor
            t = t2 - t1
            # Create a hash
            hashes.append([f1, f2, t])
    return hashes

def highpass_filter_audio(audio, sample_rate, cutoff_freq=80, order=4):
    """
    Apply a high-pass filter to audio data.
    
    Args:
        audio (numpy.ndarray): Audio time series
        sample_rate (int): Sample rate of the audio
        cutoff_freq (int): Cutoff frequency in Hz (default: 80)
        order (int): Filter order (default: 4)
    
    Returns:
        numpy.ndarray: Filtered audio data
    """
    # Normalized cutoff frequency
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design the high-pass filter
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio

