from pathlib import Path
import os

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

    # go through list, find according file
    eeg_lookup = {os.path.splitext(f)[0]: f for f in eeg_files}

    for file in psychopy_files:
        base = os.path.splitext(file)[0]
        if base in eeg_lookup:
            pairs.append((file, eeg_lookup[base]))

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
    print(pairs)

def annotate(pairs):
    '''
    input: list of pairs of files
    output: edf files that now have event event codes
    '''





def main():
    '''
    add annotations for corresponding eeg and psychopy files at the right times
    '''
    # run from weimo
    psychopy_folder = './data_collection/data'
    eeg_folder = './data_collection/eeg_data'
    annotated_eeg_folder = './data_collection/annotated_eeg'

    # for each eeg + psychopy pair
    pairs = get_pairs(psychopy_folder, eeg_folder)

    # get the time of each MI cue
    annotate(pairs)

    # add an event for each of these cues in the EEG file


if __name__ == "__main__":
    main()