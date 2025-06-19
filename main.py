import os
import numpy as np
import json
from utils.audio_processing import detect_peaks, analyze_and_visualize, hash_anchors, plot_peaks, highpass_filter_audio
import librosa
import noisereduce as nr


# Global parameters
min_distance = 10       # Minimum distance between peaks in spectrogram
hop_size = 256         # Hop size for STFT computation
cutoff = 500           # Cutoff frequency for high-pass filter
threshold = 0.0001      # Threshold for peak detection
sr_downsample = 22050  # Sample rate for downsampling audio files

# Parameters for target zone configuration
target_zone_time=200
target_zone_freq=80
fan_out=30

def fingerprintBuilder(database_folder, npy_filename="all_hashes.npy"):
    """
    Process all .wav files in the database folder to compute and save fingerprint hashes.
    
    The function analyzes each audio file by:
    1. Loading and processing the audio
    2. Detecting spectral peaks
    3. Creating anchor points and target zones
    4. Generating hash values from these points
    5. Saving all hashes to a numpy file
    
    Args:
        database_folder (str): Path to the folder containing database audio files (.wav)
        npy_filename (str): Filename to save the hash database (default: "all_hashes.npy")
    
    Returns:
        dict: Dictionary with song filename as key and list of hashes as value
    """
    global min_distance
    global hop_size
    global threshold
    global sr_downsample
    all_hashes = {}
    # Iterate through all WAV files in the database folder

    for file in os.listdir(database_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(database_folder, file)
            print(f"Processing: #{file}#")
            files=False
            if file=='pop.00015.wav':
                files=True
                print('##########################################################')
            # Load audio file with downsampling to reduce computation
            # y, sr = librosa.load(file_path,sr=sr_downsample)
            y, sr = librosa.load(file_path)
            sr_downsample=sr # Because we are not downsampling
            # Detect spectral peaks in the audio spectrogram
            coordinates, spec_shape, D = detect_peaks(y, sr, threshold_rel=threshold, min_distance=min_distance, hop_size=hop_size,files=files)

            # Generate hash values from anchor points and target zones
            anchor_dict = analyze_and_visualize(coordinates, spec_shape, D,target_zone_time=200, target_zone_freq=80, fan_out=35)
            file_hashes = hash_anchors(anchor_dict)

            # Store hashes with the associated filename
            all_hashes[file] = file_hashes

    # Save as npy file
    np.save(npy_filename, all_hashes, allow_pickle=True)
    
    print(f"Processed {len(all_hashes)} files. Total hash lists created: {len(all_hashes)}")
    return all_hashes

def audioIdentification(sample_folder, all_hashes_file="all_hashes.npy", output_file="output.txt"):
    """
    Compute hashes for query audio files and search for matches in the database.
    
    The function:
    1. Processes each query file to generate fingerprint hashes
    2. Compares these hashes with the database
    3. Calculates matching scores
    4. Returns the top matching songs / top 3 matching songs
    5. Records accuracy statistics
    6. Outputs results to specified tab-delimited text file
    
    Args:
        sample_folder (str): Path to folder containing query audio files
        all_hashes_file (str): Path to the hash database file
        output_file (str): Path to save the output results in tab-delimited format
    
    Returns:
        dict: Results with query filenames as keys and top matching songs as values
    """
    # Load the hash database
    all_hashes = np.load(all_hashes_file, allow_pickle=True).item()
    results = {}
    results_3 = {}

    # Initialize counters for accuracy metrics
    correct_matches = 0
    correct_matches_3 = 0
    total_queries = 0
    classical_correct = 0
    classical_correct_3 = 0
    classical_total = 0
    pop_correct = 0
    pop_correct_3 = 0
    pop_total = 0

    # Global parameters
    global min_distance
    global hop_size
    global cutoff
    global threshold
    global sr_downsample
    global target_zone_time
    global target_zone_freq
    global fan_out

    # Prepare for logging results
    result_log = []
    
    # Prepare for output file
    output_results = []

    # Process each query file in the sample folder
    for sample_file in os.listdir(sample_folder):
        if sample_file.endswith(".wav"):
            sample_path = os.path.join(sample_folder, sample_file)
            print(f"Processing query: {sample_file}")
            
            # Extract base name for matching validation
            sample_base_name = sample_file.split('-snippet')[0]
            # y, sr = librosa.load(file_path,sr=sr_downsample)
            y, sr = librosa.load(sample_path)
            sr_downsample=sr # Because we are not downsampling
            # reduce noise from the snippet
            # reduced_noise = nr.reduce_noise(y=y, sr=sr)
            # Pass the sninppt through an HPF
            y = highpass_filter_audio(y, sr, cutoff_freq=cutoff) 
            files=False
            if sample_file=='pop.00015-snippet-10-10.wav':
                files=True
            # Compute peaks coordinates for the sample query file
            sample_coordinates, sample_spec_shape, sample_D = detect_peaks(y, sr, threshold_rel=0.0001, hop_size=hop_size, min_distance=min_distance,files=files)
            # Create an anchor dictionary with target zones for each anchor point.
            sample_anchor_dict = analyze_and_visualize(sample_coordinates, sample_spec_shape, sample_D,
                                                        target_zone_time=target_zone_time, target_zone_freq=target_zone_freq, fan_out=fan_out)
            # Generates hashes from anchor points and their target points in the form of [f1,f2,t]
            sample_hashes = hash_anchors(sample_anchor_dict)
            sample_hash_set = set(tuple(h) for h in sample_hashes)
            
            # Compute matching score for each song
            song_scores = {}
            for song_id, file_hashes in all_hashes.items():
                file_hash_set = set(tuple(h) for h in file_hashes)
                score = len(sample_hash_set.intersection(file_hash_set))
                song_scores[song_id] = score
            # Print the scores for debugging
            print(f"Scores for {sample_file}: {song_scores}")
            
            # Get top 1 song with highest score
            top_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:1]
            results[sample_file] = top_songs
            # Print the top match for the sample file
            print(f"Top match for {sample_file}: {top_songs}")

            # Get top 3 song with highest score

            top_songs_3 = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top 3 matches for {sample_file}: {top_songs_3}")
            results_3[sample_file] = top_songs_3
            
            # Format the output line: query_file + tab + top3_matches (tab separated)
            output_line = sample_file
            for song_id, _ in top_songs_3:
                output_line += f"\t{song_id}"
                
            output_results.append(output_line)
            
            # Check if the correct song is in the top match
            total_queries += 1
            if sample_file.startswith("classical"):
                classical_total += 1
            elif sample_file.startswith("pop"):
                pop_total += 1
            
            # Print the matches for top match and top 3 matches 
            print(sample_base_name,song_id,top_songs)
            if any(sample_base_name in song_id for song_id, _ in top_songs):
                correct_matches += 1
                print('MATCHED!!!!!!!!!')
                if sample_file.startswith("classical"):
                    classical_correct += 1
                elif sample_file.startswith("pop"):
                    pop_correct += 1
            print(f"Top match for {sample_file}:")
            for song_id, score in top_songs:
                print(f"{song_id}: {score} matching hashes(score)")
            print("-" * 40)
            
            if any(sample_base_name in song_id for song_id, _ in top_songs_3):
                correct_matches_3 += 1
                print('MATCHED IN TOP 3!!!!!!!!!')
                if sample_file.startswith("classical"):
                    classical_correct_3 += 1
                elif sample_file.startswith("pop"):
                    pop_correct_3 += 1
            print(f"Top 3 match for {sample_file}:")
            for song_id, score in top_songs_3:
                print(f"{song_id}: {score} matching hashes(score)")
            print("-" * 40)

    # Calculate accuracy metrics
    accuracy = (correct_matches / total_queries) * 100 if total_queries > 0 else 0
    classical_accuracy = (classical_correct / classical_total) * 100 if classical_total > 0 else 0
    pop_accuracy = (pop_correct / pop_total) * 100 if pop_total > 0 else 0
    
    accuracy_3 = (correct_matches_3 / total_queries) * 100 if total_queries > 0 else 0
    classical_accuracy_3 = (classical_correct_3 / classical_total) * 100 if classical_total > 0 else 0
    pop_accuracy_3 = (pop_correct_3 / pop_total) * 100 if pop_total > 0 else 0
    
    # Append results and parameters to the result_log
    result_log.append(f"Parameters: hop_size={hop_size}, cutoff={cutoff}, min_distance={min_distance},threshold={threshold},sr={sr_downsample},target_zone_time={target_zone_time}, target_zone_freq={target_zone_freq}, fan_out={fan_out}")
    result_log.append(f"Top-1 Results:")
    result_log.append(f"Overall Accuracy: {accuracy:.2f}% ({correct_matches}/{total_queries} correct matches)")
    result_log.append(f"Classical Accuracy: {classical_accuracy:.2f}% ({classical_correct}/{classical_total} correct matches)")
    result_log.append(f"Pop Accuracy: {pop_accuracy:.2f}% ({pop_correct}/{pop_total} correct matches)")
    result_log.append(f"Top-3 Results:")
    result_log.append(f"Overall Accuracy: {accuracy_3:.2f}% ({correct_matches_3}/{total_queries} correct matches)")
    result_log.append(f"Classical Accuracy: {classical_accuracy_3:.2f}% ({classical_correct_3}/{classical_total} correct matches)")
    result_log.append(f"Pop Accuracy: {pop_accuracy_3:.2f}% ({pop_correct_3}/{pop_total} correct matches)")
    
    # Write to result.txt
    with open("result.txt", "a") as result_file:
        result_file.write("\n".join(result_log) + "\n")
    
    # Write to output file in the specified format
    with open(output_file, "w") as out_file:
        out_file.write("\n".join(output_results))
    
    return results

if __name__ == "__main__":
    # Set folder paths and filenames
    database_folder = "dataset/testing/"
    sample_folder = r"dataset/query_recordings/"
    
    # Create hashes for all songs in the database
    fingerprintBuilder(database_folder, npy_filename="all_hashes.npy")
    
    # Search for the sample in the database and return top matches
    search_results = audioIdentification(sample_folder, all_hashes_file="all_hashes.npy")