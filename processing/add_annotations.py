from pathlib import Path
import os
import mne
import pandas as pd
from datetime import datetime, date
import math


def get_filenames(folder_name):
    '''
    input: folder name
    output: list of filepaths
    '''
    directory = Path(folder_name)
    filenames = [p.name for p in directory.iterdir() if p.is_file()]
    return filenames

def get_pairs_util(psychopy_files, eeg_files):
    '''
    input: a list of psychopy files and a list of eeg files
    output: a list of 2 tuples of them with their approprite pairs
    '''
    # format of psychopy files is: [firstname][date].csv
    # format of eeg files is: [firstname][date].edf
    # since we only do each person once a day max, they will always be unique
    pairs = []

    eeg_edf_lookup = {os.path.splitext(f)[0]: f for f in eeg_files if f.endswith('.edf')}
    eeg_csv_lookup = {os.path.splitext(f)[0]: f for f in eeg_files if f.endswith('.csv')}

    for file in psychopy_files:
        base = os.path.splitext(file)[0].removesuffix('pp')
        if base in eeg_edf_lookup and base in eeg_csv_lookup:
            pairs.append((file, eeg_edf_lookup[base], eeg_csv_lookup[base]))

    return pairs



def get_pairs(psychopy_folder, eeg_folder):
    '''
    input: n_amount of files to do, if None then do as many as possible
    output: a list of file path pairs
    '''
    # get file names of the psychopy folder
    psychopy_files = get_filenames(psychopy_folder)

    # get file names of eeg_data folder
    eeg_files = get_filenames(eeg_folder)

    # make pairs according to naming conventions
    pairs = get_pairs_util(psychopy_files, eeg_files)
    return pairs

def get_time(lines):
    '''
    input: a list of meta data lines
    output: a time
    '''
    for line in lines:
        if line.startswith('# Time ='):
            time_str = line.split(', ')[1].split(',')[0].strip()
            break
    
    time = datetime.strptime(time_str, '%H:%M:%S.%f').time()

    return time

def get_psycho_time(df):
    '''
    input: path to a psychopy time
    output: a time
    '''
    start_col = df['expStart']
    start_time = start_col[0]
    start = datetime.strptime(start_time, '%Y-%m-%d %Hh%M.%S.%f %z').time()
    return start

def clean_time_values(values):
    
    '''
    clean up data to ensure only numbers are counted
    '''
    
    cleaned = []
    for x in values:
        if isinstance(x, (int, float)) and not math.isnan(x):
            cleaned.append(float(x))
    return cleaned

def annotate(pairs, output_folder, eeg_folder, psychopy_folder):
    '''
    input: list of pairs of files
    output: fif files that now have event event codes
    '''

    for pair in pairs:
        eeg_file = pair[1]
        eeg_info = pair[2]
        psycho_file = pair[0]
        # load in the raw
        raw = mne.io.read_raw_edf(f'{eeg_folder}/{eeg_file}', preload=False)
        # get the start time for the eeg data and the psychopy data
        with open(f'{eeg_folder}/{eeg_info}') as f:
            lines = f.readlines()
        
        # establish eeg time
        eeg_time = get_time(lines)
        # establish psychopy time
        df = pd.read_csv(f'{psychopy_folder}/{psycho_file}')
        psychopy_time = get_psycho_time(df)


        # get stimulus onset times for movement
        # TODO: times for jaw clenches
        movement_times = df['go_txt.started'].tolist()
        clean_move_times = [x for x in movement_times if not math.isnan(x)]

        jaw_time = []
        if 'jaw_txt.started' in df.columns:
            jaw_clench = df['jaw_txt.started'].tolist()
            clean_jaw = clean_time_values(jaw_clench)
            #make sure to change label to correct one once integrated 
        else:
            clean_jaw = []

        # get difference of times, eeg will always start sooner
        diff = datetime.combine(date.min, psychopy_time) - datetime.combine(date.min, eeg_time)
        seconds_diff = diff.total_seconds()

        # add seconds to all elements of psychopy_time
        move_onsets = [x + seconds_diff for x in clean_move_times]
        move_durations = [4.0] * len(move_onsets)
        move_desc = ['move'] * len(move_onsets)

        jaw_onset = [x + seconds_diff for x in clean_jaw]
        jaw_duration = [1.0] * len(jaw_onset) #change time according to how long jaw clench is 
        jaw_desc = ['jaw_clench'] * len(jaw_onset)

        #combine move and jaw onsets 
        onsets = move_onsets + jaw_onset
        durations = move_durations + jaw_duration
        descriptions = move_desc + jaw_desc

        # set annotations
        annotation = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(annotation)
        '''
        if raw.annotations is not None and len(raw.annotations) > 0:
            raw.set_annotations(raw.annotations + annotation)
        else:
            raw.set_annotations(annotation)
        '''
        #labeling the events in the raw data 
        events, event_id = mne.events_from_annotations(raw)
        print(f"[INFO] {eeg_file}: event_id={event_id}, #_events={len(events)}")
        eeg_file = eeg_file[:-4] # removes the .edf suffix

        # save
        raw.save(f'{output_folder}/{eeg_file}.fif', overwrite=True)


def main():
    '''
    add annotations for corresponding eeg and psychopy files at the right times
    '''
    # run from weimo
    # should be named "namedate(month then day).fif"
    psychopy_folder = './data_collection/data' 
    eeg_folder = './data_collection/eeg_data'
    annotated_eeg_folder = './data_collection/annotated_eeg'

    # for each eeg + psychopy pair
    pairs = get_pairs(psychopy_folder, eeg_folder)
    print(pairs)

    # get the time of each MI cue
    annotate(pairs=pairs, output_folder=annotated_eeg_folder, 
             eeg_folder=eeg_folder, psychopy_folder=psychopy_folder)


if __name__ == "__main__":
    main()
    