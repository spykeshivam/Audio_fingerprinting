import os
import numpy as np
import json
from utils.audio_processing import detect_peaks, analyze_and_visualize, hash_anchors,plot_peaks, highpass_filter_audio
import librosa

min_distance=10
hop_size=256
cutoff=250

def create_hash(database_folder, npy_filename="all_hashes.npy", json_filename="all_hashes.json"):
    """
    Processes all .wav files in the database_folder to compute and save hashes.
    
    Returns:
        all_hashes (dict): Dictionary with song filename as key and list of hashes as value.
    """
    global min_distance
    global hop_size
    all_hashes = {}
    for file in os.listdir(database_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(database_folder, file)
            print(f"Processing: {file}")
            y, sr = librosa.load(file_path)
            coordinates, spec_shape, D = detect_peaks(y,sr, threshold_rel=0.01,min_distance=min_distance,hop_size=hop_size)
            anchor_dict = analyze_and_visualize(coordinates, spec_shape, D,
                                                target_zone_time=200, target_zone_freq=80, fan_out=35)
            file_hashes = hash_anchors(anchor_dict)
            all_hashes[file] = file_hashes

    # Save as npy file
    np.save(npy_filename, all_hashes, allow_pickle=True)
    
    # Save as JSON file (convert NumPy types to native Python types)
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(json_filename, "w") as f:
        json.dump(all_hashes, f, default=convert_numpy)
    
    print(f"Processed {len(all_hashes)} files. Total hash lists created: {len(all_hashes)}")
    return all_hashes

def search_hash_file(sample_file, all_hashes_file="all_hashes.npy", target_zone_time=200, target_zone_freq=80, fan_out=50):
    """
    Computes the hash for a sample query file, then searches the database for matching songs.
    
    Returns:
        top_songs (list): Top 10 songs with highest matching hash scores.
    """
    # Compute hashes for the sample query file
    y, sr = librosa.load(sample_file)
    sample_coordinates, sample_spec_shape, sample_D = detect_peaks(y,sr, threshold_rel=0.0001,min_distance=10)
    #print(sample_coordinates)
    #plot_peaks(sample_D,sample_coordinates)
    sample_anchor_dict = analyze_and_visualize(sample_coordinates, sample_spec_shape, sample_D,
                                                target_zone_time=target_zone_time, target_zone_freq=target_zone_freq, fan_out=fan_out,)
    sample_hashes = hash_anchors(sample_anchor_dict)
    sample_hash_set = set(tuple(h) for h in sample_hashes)
    
    # Load all hashes from the saved file
    all_hashes = np.load(all_hashes_file, allow_pickle=True).item()
    
    # Compute matching score for each song
    song_scores = {}
    for song_id, file_hashes in all_hashes.items():
        file_hash_set = set(tuple(h) for h in file_hashes)
        score = len(sample_hash_set.intersection(file_hash_set))
        song_scores[song_id] = score
    
    # Get top 10 songs with highest score
    top_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 matching songs:")
    for song_id, score in top_songs:
        print(f"{song_id}: {score} matching hashes(score)")
    
    return top_songs

def search_hash(sample_folder, all_hashes_file="all_hashes.npy", target_zone_time=200, target_zone_freq=80, fan_out=50):
    """
    Computes hashes for all sample query files in a folder, then searches the database for matching songs.
    Returns:
        results (dict): Dictionary where keys are query filenames and values are top 1 matching song.
    """
    all_hashes = np.load(all_hashes_file, allow_pickle=True).item()
    results = {}
    results_3 = {}
    correct_matches = 0
    correct_matches_3 = 0
    total_queries = 0
    classical_correct = 0
    classical_correct_3 = 0
    classical_total = 0
    pop_correct = 0
    pop_correct_3 = 0
    pop_total = 0
    global min_distance
    global hop_size
    global cutoff

    for sample_file in os.listdir(sample_folder):
        if sample_file.endswith(".wav"):
            sample_path = os.path.join(sample_folder, sample_file)
            print(f"Processing query: {sample_file}")
            
            # Extract base name for matching validation
            sample_base_name = sample_file.split('-snippet')[0]  # Extracts 'pop.00011' from 'pop.00011-snippet-10-20.wav'
            y, sr = librosa.load(sample_path)
            y=highpass_filter_audio(y, sr, cutoff_freq=cutoff)
            # Compute hashes for the sample query file
            sample_coordinates, sample_spec_shape, sample_D = detect_peaks(y,sr, threshold_rel=0.0001,hop_size=hop_size,min_distance=min_distance)
            sample_anchor_dict = analyze_and_visualize(sample_coordinates, sample_spec_shape, sample_D,
                                                        target_zone_time=target_zone_time, target_zone_freq=target_zone_freq, fan_out=fan_out)
            sample_hashes = hash_anchors(sample_anchor_dict)
            sample_hash_set = set(tuple(h) for h in sample_hashes)
            
            # Compute matching score for each song
            song_scores = {}
            for song_id, file_hashes in all_hashes.items():
                file_hash_set = set(tuple(h) for h in file_hashes)
                score = len(sample_hash_set.intersection(file_hash_set))
                song_scores[song_id] = score
            
            # Get top 1 song with highest score
            top_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:1]
            results[sample_file] = top_songs

            # Get top 3 song with highest score
            top_songs_3 = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            results_3[sample_file] = top_songs_3
            
            # Check if the correct song is in the top match
            total_queries += 1
            if sample_file.startswith("classical"):
                classical_total += 1
            elif sample_file.startswith("pop"):
                pop_total += 1
            
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
    
    accuracy = (correct_matches / total_queries) * 100 if total_queries > 0 else 0
    classical_accuracy = (classical_correct / classical_total) * 100 if classical_total > 0 else 0
    pop_accuracy = (pop_correct / pop_total) * 100 if pop_total > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_matches}/{total_queries} correct matches)")
    print(f"Classical Accuracy: {classical_accuracy:.2f}% ({classical_correct}/{classical_total} correct matches)")
    print(f"Pop Accuracy: {pop_accuracy:.2f}% ({pop_correct}/{pop_total} correct matches)")

    accuracy_3 = (correct_matches_3 / total_queries) * 100 if total_queries > 0 else 0
    classical_accuracy_3 = (classical_correct_3 / classical_total) * 100 if classical_total > 0 else 0
    pop_accuracy_3 = (pop_correct_3 / pop_total) * 100 if pop_total > 0 else 0
    
    print(f"Overall Accuracy: {accuracy_3:.2f}% ({correct_matches_3}/{total_queries} correct matches)")
    print(f"Classical Accuracy: {classical_accuracy_3:.2f}% ({classical_correct_3}/{classical_total} correct matches)")
    print(f"Pop Accuracy: {pop_accuracy_3:.2f}% ({pop_correct_3}/{pop_total} correct matches)")
    
    return results

if __name__ == "__main__":
    # Set folder paths and filenames
    database_folder = "dataset/database_recordings/database_recordings/"
    sample_file = r"dataset/query_recordings/query_recordings/classical.00079-snippet-10-10.wav"
    #sample_file = r"highpass_filtered_music.wav"
    # Create hashes for all songs in the database
    #create_hash(database_folder, npy_filename="all_hashes.npy", json_filename="all_hashes.json")
    
    # Search for the sample in the database and return top matches
    #search_hash_file(sample_file, all_hashes_file="all_hashes.npy", target_zone_time=200, target_zone_freq=80, fan_out=10)
    sample_folder = r"dataset/query_recordings/query_recordings/"
    search_results = search_hash(sample_folder, all_hashes_file="all_hashes.npy", target_zone_time=200, target_zone_freq=80, fan_out=50)