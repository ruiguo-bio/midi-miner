from __future__ import print_function

import pretty_midi

import numpy as np
import pickle
import pandas as pd
import scipy.stats
from collections import Counter
from functools import reduce
import os
from copy import deepcopy
import sys
import argparse
import json
import logging
import coloredlogs



def remove_empty_track(midi_file):
    '''

    1. remove emtpy track,
    also remove track with fewer than 10% notes of the track
    with most notes

    ********
    Return: pretty_midi object
    '''

    try:
        pretty_midi_data = pretty_midi.PrettyMIDI(midi_file)

    except Exception as e:
        logger.warning(f'exceptions {e} when read the file {midi_file}')

        return None, None

    note_count = np.array([len(instrument.notes) \
                                for instrument in pretty_midi_data.instruments])
                  #
    if len(pretty_midi_data.instruments) > 3:
        empty_indices = np.array(note_count / np.sort(note_count)[-2] < 0.1)
    else:
        empty_indices = np.array(note_count / np.max(note_count) < 0.1)

    for i, instrument in enumerate(pretty_midi_data.instruments):
        all_less_than_10 = True
        for note in instrument.notes:
            if note.pitch > 10:
                all_less_than_10 = False
        if all_less_than_10:
            empty_indices[i] = True


    if np.sum(empty_indices) > 0:
        for index in sorted(np.where(empty_indices)[0],reverse=True):

            del pretty_midi_data.instruments[index]


    return pretty_midi_data

def remove_duplicate_tracks(features, replace=False):
    if not replace:
        features = features.copy()

    file_names = features.file_names.unique()
    duplicates = []

    for file_name in file_names:
        file_features = features[features.file_names == file_name]
        number_notes = Counter(file_features.num_notes)
        notes = []
        for ele in number_notes:
            if number_notes[ele] > 1:
                notes.append(ele)
        h_pits = []
        for note in notes:
            number_h_pit = Counter(file_features[file_features.num_notes == note].h_pit)

            for ele in number_h_pit:
                if number_h_pit[ele] > 1:
                    h_pits.append(ele)

        l_pits = []
        for h_pit in h_pits:
            number_l_pit = Counter(file_features[file_features.h_pit == h_pit].l_pit)

            for ele in number_l_pit:
                if number_l_pit[ele] > 1:
                    l_pits.append(ele)

        notes = list(set(notes))
        h_pits = list(set(h_pits))
        l_pits = list(set(l_pits))

        for note in notes:
            note_index = file_features[file_features.num_notes == note].index.values
            for h_pit in h_pits:
                h_pit_index = file_features[file_features.h_pit == h_pit].index.values
                for l_pit in l_pits:
                    l_pit_index = file_features[file_features.l_pit == l_pit].index.values

                    index_intersect = reduce(np.intersect1d, (note_index, h_pit_index, l_pit_index))

                    if len(index_intersect) > 1:
                        duplicates.append(index_intersect)

    ### copy the labels in the tracks to be removed

    melody_track_name = ['sing', 'vocals', 'vocal', 'melody', 'melody:']
    bass_track_name = ['bass', 'bass:']
    chord_track_name = ['chord', 'chords', 'harmony']
    drum_track_name = ['drum', 'drums']

    for indices in duplicates:
        melody_track = False
        bass_track = False
        chord_track = False
        drum_track = False
        labels = features.loc[indices, 'trk_names']
        for label in labels:
            if label in melody_track_name:
                melody_track = True

            elif label in bass_track_name:

                bass_track = True

            elif label in chord_track_name:
                chord_track = True
            elif label in drum_track_name:
                drum_track = True
            else:
                pass

        if melody_track:
            features.loc[indices, 'trk_names'] = 'melody'
        if bass_track:
            features.loc[indices, 'trk_names'] = 'bass'
        if chord_track:
            features.loc[indices, 'trk_names'] = 'chord'
        if drum_track:
            features.loc[indices, 'trk_names'] = 'drum'


        features.drop(indices[1:], inplace=True)
        logger.info(indices[1:])

    return features


def remove_file_duplicate_tracks(features, pm):
    duplicates = []

    index_to_remove = []
    number_notes = Counter(features.num_notes)
    notes = []
    for ele in number_notes:
        if number_notes[ele] > 1:
            notes.append(ele)
    h_pits = []
    for note in notes:
        number_h_pit = Counter(features[features.num_notes == note].h_pit)

        for ele in number_h_pit:
            if number_h_pit[ele] > 1:
                h_pits.append(ele)

    l_pits = []
    for h_pit in h_pits:
        number_l_pit = Counter(features[features.h_pit == h_pit].l_pit)

        for ele in number_l_pit:
            if number_l_pit[ele] > 1:
                l_pits.append(ele)

    notes = list(set(notes))
    h_pits = list(set(h_pits))
    l_pits = list(set(l_pits))

    for note in notes:
        note_index = features[features.num_notes == note].index.values
        for h_pit in h_pits:
            h_pit_index = features[features.h_pit == h_pit].index.values
            for l_pit in l_pits:
                l_pit_index = features[features.l_pit == l_pit].index.values

                index_intersect = reduce(np.intersect1d, (note_index, h_pit_index, l_pit_index))

                if len(index_intersect) > 1:
                    duplicates.append(index_intersect)

    ### copy the labels in the tracks to be removed

    melody_track_name = ['sing', 'vocals', 'vocal', 'melody', 'melody:']
    bass_track_name = ['bass', 'bass:']
    chord_track_name = ['chord', 'chords', 'harmony']
    drum_track_name = ['drum', 'drums']

    for indices in duplicates:
        melody_track = False
        bass_track = False
        chord_track = False
        drum_track = False
        labels = features.loc[indices, 'trk_names']
        for label in labels:
            if label in melody_track_name:
                melody_track = True

            elif label in bass_track_name:

                bass_track = True

            elif label in chord_track_name:
                chord_track = True

            elif label in drum_track_name:
                drum_track = True

            else:
                pass

        if melody_track:
            features.loc[indices, 'trk_names'] = 'melody'
        if bass_track:
            features.loc[indices, 'trk_names'] = 'bass'
        if chord_track:
            features.loc[indices, 'trk_names'] = 'chord'

        if drum_track:
            features.loc[indices, 'trk_names'] = 'drum'


        features.drop(indices[1:], inplace=True)
        # logger.info(f'indices are {indices}')


        for index in indices[1:]:
            # logger.info(f'index is {index}')
            index_to_remove.append(index)

    indices = np.sort(np.array(index_to_remove))

    for index in indices[::-1]:
        # logger.info(f'index is {index}')
        del pm.instruments[index]

    features.reset_index(inplace=True, drop='index')

    return

def walk(folder_name):
    files = []
    for p, d, f in os.walk(folder_name):
        for file in f:
            endname = file.split('.')[-1].lower()
            if endname == 'mid' or endname == 'midi':
                files.append(os.path.join(p,file))
    return files


def relative_duration(pm):
    notes = np.array([len(pm.instruments[i].notes) for i in range(len(pm.instruments))])
    if np.max(notes) == 0:
        return None
    relative_durations = notes / np.max(notes)

    relative_durations = relative_durations[:, np.newaxis]

    assert relative_durations.shape == (len(pm.instruments), 1)

    return relative_durations


def number_of_notes(pm):
    '''
    read pretty-midi data
    '''
    number_of_notes = []
    for instrument in pm.instruments:
        number_of_notes.append(len(instrument.notes))

    number_of_notes = np.array(number_of_notes, dtype='uint16')

    number_of_notes = number_of_notes[:, np.newaxis]

    assert number_of_notes.shape == (len(pm.instruments), 1)

    return number_of_notes


def occupation_polyphony_rate(pm):
    occupation_rate = []
    polyphony_rate = []

    for instrument in pm.instruments:
        piano_roll = get_piano_roll(instrument)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(np.count_nonzero(np.any(piano_roll, 0)) / piano_roll.shape[1])
        if np.count_nonzero(np.any(piano_roll, 0)) == 0:
            polyphony_rate.append(0)
        else:
            polyphony_rate.append(
                np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(np.any(piano_roll, 0)))

    occupation_rate = np.array(occupation_rate)
    zero_idx = np.where(occupation_rate < 0.01)[0]
    if len(zero_idx) > 0:
        occupation_rate[zero_idx] = 0

    occupation_rate = occupation_rate[:, np.newaxis]


    polyphony_rate = np.array(polyphony_rate)
    zero_idx = np.where(polyphony_rate < 0.01)[0]
    if len(zero_idx) > 0:
        polyphony_rate[zero_idx] = 0

    polyphony_rate = polyphony_rate[:, np.newaxis]

    return occupation_rate, polyphony_rate

def pitch(pm):
    '''
    read pretty midi data

    Returns
        -------
        a numpy array in the shape of (number of tracks, 8)

        the 8 columns are highest pitch, lowest pitch, pitch mode, pitch std,
        and the norm value across different tracks for those values

    '''

    highest = []
    lowest = []
    modes = []
    stds = []

    def array_creation_by_count(counts):
        result = []
        for i, count in enumerate(counts):
            if count != 0:
                result.append([i] * count)

        result = np.array([item for sublist in result for item in sublist])
        return result

    for track in pm.instruments:
        highest_note = np.where(np.any(get_piano_roll(track), 1))[0][-1]
        lowest_note = np.where(np.any(get_piano_roll(track), 1))[0][0]
        pitch_array = array_creation_by_count(np.count_nonzero(get_piano_roll(track), 1))

        mode_pitch = scipy.stats.mode(pitch_array)
        mode_pitch = mode_pitch.mode[0]

        # logger.info(mode_pitch)

        std_pitch = np.std(pitch_array)

        # logger.info(std_pitch)

        highest.append(highest_note)
        lowest.append(lowest_note)
        modes.append(mode_pitch)
        stds.append(std_pitch)

    highest = np.array(highest, dtype='uint8')
    lowest = np.array(lowest, dtype='uint8')
    modes = np.array(modes, dtype='uint8')
    stds = np.array(stds, dtype='float32')


    if np.max(highest) - np.min(highest) == 0:
        highest_norm = np.ones_like(highest)
    else:

        highest_norm = (highest - np.min(highest)) / (np.max(highest) - np.min(highest))

    if np.max(lowest) - np.min(lowest) == 0:
        lowest_norm = np.zeros_like(lowest)
    else:
        lowest_norm = (lowest - np.min(lowest)) / (np.max(lowest) - np.min(lowest))

    if np.max(modes) - np.min(modes) == 0:
        modes_norm = np.zeros_like(modes)
    else:
        modes_norm = (modes - np.min(modes)) / (np.max(modes) - np.min(modes))

    if np.max(stds) - np.min(stds) == 0:
        stds_norm = np.zeros_like(stds)
    else:
        stds_norm = (stds - np.min(stds)) / (np.max(stds) - np.min(stds))

    result = np.vstack((highest, lowest, modes, stds, highest_norm, lowest_norm, modes_norm, stds_norm))
    result = result.T

    # logger.info(result.shape)
    assert result.shape == (len(pm.instruments), 8)

    return result


def pitch_intervals(pm):
    '''
    use pretty-midi data here

     Returns
        -------
        a numpy array in the shape of (number of tracks, 5)

        the 5 columns are number of different intervals, largest interval,
        smallest interval, mode interval and interval std of this track,
        and the norm value across different tracks for those values

    '''

    different_interval = []
    largest_interval = []
    smallest_interval = []
    mode_interval = []
    std_interval = []

    def get_intervals(notes, threshold=-1):
        '''

        threshold is the second for the space between two consecutive notes
        '''

        intervals = []
        for i in range(len(notes) - 1):
            note1 = notes[i]
            note2 = notes[i + 1]

            if note1.end - note2.start >= threshold:
                if note2.end >= note1.end:

                    intervals.append(abs(note2.pitch - note1.pitch))
        return np.array(intervals)

    for instrument in pm.instruments:
        intervals = get_intervals(instrument.notes, -3)
        #         logger.info(f'intervals is {intervals}')

        if len(intervals) > 0:

            different_interval.append(len(np.unique(intervals)))
            largest_interval.append(np.max(intervals))
            smallest_interval.append(np.min(intervals))
            mode_interval.append(scipy.stats.mode(intervals).mode[0])
            std_interval.append(np.std(intervals))
        else:
            different_interval.append(-1)
            largest_interval.append(-1)
            smallest_interval.append(-1)
            mode_interval.append(-1)
            std_interval.append(-1)

    different_interval = np.array(different_interval, dtype='uint8')
    largest_interval = np.array(largest_interval, dtype='uint8')
    smallest_interval = np.array(smallest_interval, dtype='uint8')
    mode_interval = np.array(mode_interval, dtype='uint8')
    std_interval = np.array(std_interval, dtype='float32')


    if np.max(different_interval) - np.min(different_interval) == 0:
        different_interval_norm = np.zeros_like(different_interval)
    else:
        different_interval_norm = (different_interval - np.min(different_interval)) / (
                    np.max(different_interval) - np.min(different_interval))

    if np.max(largest_interval) - np.min(largest_interval) == 0:
        largest_interval_norm = np.ones_like(largest_interval)
    else:
        largest_interval_norm = (largest_interval - np.min(largest_interval)) / (
                    np.max(largest_interval) - np.min(largest_interval))

    if np.max(smallest_interval) - np.min(smallest_interval) == 0:
        smallest_interval_norm = np.zeros_like(smallest_interval)
    else:
        smallest_interval_norm = (smallest_interval - np.min(smallest_interval)) / (
                    np.max(smallest_interval) - np.min(smallest_interval))

    if np.max(mode_interval) - np.min(mode_interval) == 0:
        mode_interval_norm = np.zeros_like(mode_interval)
    else:
        mode_interval_norm = (mode_interval - np.min(mode_interval)) / (np.max(mode_interval) - np.min(mode_interval))

    if np.max(std_interval) - np.min(std_interval) == 0:
        std_interval_norm = np.zeros_like(std_interval)
    else:
        std_interval_norm = (std_interval - np.min(std_interval)) / (np.max(std_interval) - np.min(std_interval))



    result = np.vstack((different_interval, largest_interval, smallest_interval, \
                        mode_interval, std_interval, different_interval_norm, \
                        largest_interval_norm, smallest_interval_norm, \
                        mode_interval_norm, std_interval_norm))

    result = result.T

    assert (result.shape == (len(pm.instruments), 10))

    return result



def note_durations(pm):
    '''
    use pretty-midi data here

    Parameters
        ----------
        data : pretty-midi data

         Returns
        -------
        a numpy array in the shape of (number of tracks, 4)

        the 4 columns are longest, shortest, mean, std of note durations
        and the norm value across different tracks for those values
    '''

    longest_duration = []
    shortest_duration = []
    mean_duration = []
    std_duration = []

    for instrument in pm.instruments:
        notes = instrument.notes
        durations = np.array([note.end - note.start for note in notes])

        longest_duration.append(np.max(durations))
        shortest_duration.append(np.min(durations))
        mean_duration.append(np.mean(durations))
        std_duration.append(np.std(durations))


    longest_duration = np.array(longest_duration)
    shortest_duration = np.array(shortest_duration)
    mean_duration = np.array(mean_duration)
    std_duration = np.array(std_duration)

    if np.max(longest_duration) - np.min(longest_duration) == 0:
        longest_duration_norm = np.ones_like(longest_duration)
    else:
        longest_duration_norm = (longest_duration - np.min(longest_duration)) / (
                    np.max(longest_duration) - np.min(longest_duration))

    if np.max(shortest_duration) - np.min(shortest_duration) == 0:
        shortest_duration_norm = np.zeros_like(shortest_duration)
    else:
        shortest_duration_norm = (shortest_duration - np.min(shortest_duration)) / (
                    np.max(shortest_duration) - np.min(shortest_duration))

    if np.max(mean_duration) - np.min(mean_duration) == 0:
        mean_duration_norm = np.zeros_like(mean_duration)
    else:
        mean_duration_norm = (mean_duration - np.min(mean_duration)) / (np.max(mean_duration) - np.min(mean_duration))

    if np.max(std_duration) - np.min(std_duration) == 0:
        std_duration_norm = np.zeros_like(std_duration)
    else:
        std_duration_norm = (std_duration - np.min(std_duration)) / (np.max(std_duration) - np.min(std_duration))

    result = np.vstack((longest_duration, shortest_duration, mean_duration, \
                        std_duration, longest_duration_norm, shortest_duration_norm, \
                        mean_duration_norm, std_duration_norm))

    result = result.T

    # logger.info(result.shape)

    assert result.shape == (len(pm.instruments), 8)
    return result


def get_piano_roll(track,fs=100):
    """Compute a piano roll matrix of this instrument.

    Parameters
    ----------
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.


    Returns
    -------
    piano_roll : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument.

    """
    # If there are no notes, return an empty matrix
    if track.notes == []:
        return np.array([[]]*128)
    # Get the end time of the last event
    end_time = track.get_end_time()
    # Extend end time if one was provided

    # Allocate a matrix of zeros - we will add in as we go
    piano_roll = np.zeros((128, int(fs*end_time)))

    # Add up piano roll matrix, note-by-note
    for note in track.notes:
        # Should interpolate
        piano_roll[note.pitch,
                   int(note.start*fs):int(note.end*fs)] += note.velocity

    return piano_roll



def cal_file_features(midi_file):
    '''
    compute 34 features from midi data. Each track of each song have 30 features

    1 set of feature:
    duration, number of notes, occupation rate, polyphony rate,

    2 set of feature:
    Highest pitch, lowest pitch, pitch mode, pitch std,
    Highest pitch norm, lowest pitch norm, pitch mode norm, pitch std norm

    3 set of feature

    number of interval, largest interval,
    smallest interval, interval mode,
    number of interval norm, largest interval norm,
    smallest interval norm, interval mode norm

    4 set of feature

    longest note duration, shortest note duration,
     mean note duration, note duration std,
     longest note duration norm, shortest note duration norm,
     mean note duration norm, note duration std norm

    for all the normed feature,  it is the normalised features
    across different tracks within a midi file

    5 set of feature:
    track_programs,track_names,file_names,is_drum

    '''


    pm = remove_empty_track(midi_file)

    if pm is None or len(pm.instruments) == 0:
        return None,None

    for track in pm.instruments:
        if np.any(get_piano_roll(track)) == False:
            return None,None


    track_programs = np.array([i.program for i in pm.instruments])[:, np.newaxis]
    track_names = []

    try:
        for instrument in pm.instruments:
            if len(instrument.name.rsplit()) > 0:
                track_names.append(instrument.name.rsplit()[0].lower())
            #             if instrument.name.strip() is not '':
            #                 track_names.append(instrument.name.rsplit()[0].lower())
            else:
                track_names.append('')

    except Exception as e:
        logger.warning(e)
        return None,None

    #     basename = os.path.basename(midi_file)
    #     pm.write('/Users/ruiguo/Downloads/2000midi/new/' + basename)

    track_names = np.array(track_names)[:, np.newaxis]
    file_names = np.array([midi_file] * len(pm.instruments))[:, np.newaxis]
    is_drum = np.array([i.is_drum for i in pm.instruments])[:, np.newaxis]

    rel_durations = relative_duration(pm)
    if rel_durations is None:
        logger.warining(f'no notes in file {midi_file}')
        return None,None
    number_notes = number_of_notes(pm)
    occup_rate,poly_rate = occupation_polyphony_rate(pm)
    pitch_features = pitch(pm)
    pitch_interval_features = pitch_intervals(pm)
    note_duration_features = note_durations(pm)


    all_features = np.hstack((track_programs, track_names, file_names, is_drum, \
                              rel_durations, number_notes, occup_rate, \
                              poly_rate, pitch_features, \
                              pitch_interval_features, note_duration_features
                              ))

    # logger.info(all_features.shape)
    assert all_features.shape == (len(pm.instruments), 34)

    return all_features, pm


melody_track_name = ['sing','vocals','vocal','melody','melody:']
bass_track_name = ['bass','bass:']
chord_track_name = ['chord','chords','harmony']
drum_track_name = ['drum','drums']
check_melody = lambda x: x in melody_track_name
check_bass = lambda x: x in bass_track_name
check_chord = lambda x: x in chord_track_name
check_drum = lambda x: x in drum_track_name

columns=['trk_prog','trk_names','file_names','is_drum',
         'dur', 'num_notes', 'occup_rate', 'poly_rate',
   'h_pit', 'l_pit', 'pit_mode', 'pit_std',
         'h_pit_nor', 'l_pit_nor', 'pit_mode_nor', 'pit_std_nor',
 'num_intval', 'l_intval', 's_intval', 'intval_mode', 'intval_std',
'num_intval_nor', 'l_intval_nor',  's_intval_nor','intval_mode_nor','intval_std_nor',
         'l_dur','s_dur', 'mean_dur', 'dur_std',
         'l_dur_nor','s_dur_nor', 'mean_dur_nor', 'dur_std_nor']

boolean_dict = {'True': True, 'False': False}

def add_labels(features):
    features = pd.DataFrame(features, columns=columns)

    for name in columns[4:]:
        features[name] = pd.to_numeric(features[name])

    features['trk_prog'] = pd.to_numeric(features['trk_prog'])
    features['is_drum'] = features['is_drum'].map(boolean_dict)

    return features


def predict_labels(features, melody_model, bass_model, chord_model,drum_model):
    temp_features = features.copy()
    temp_features = temp_features.drop(temp_features.columns[:4], axis=1)

    predicted_melody = melody_model.predict(temp_features)
    predicted_bass = bass_model.predict(temp_features)
    predicted_chord = chord_model.predict(temp_features)
    predicted_drum = drum_model.predict(temp_features)



    for index,value in enumerate(predicted_melody):
        if value:
            if features.iloc[index]['poly_rate'] > 0.3:
                predicted_melody[index] = False

    for index,value in enumerate(predicted_bass):
        if value:
            if features.iloc[index]['poly_rate'] > 0.3:
                predicted_bass[index] = False




    if np.sum(predicted_melody) == 0:
        melody_candidates = features[(features.mean_dur < 1) & (features.occup_rate > 0.6) & (features.poly_rate < 0.1)].index.values
        if np.sum(predicted_bass) > 0:
            bass_index = np.where(predicted_bass)[0][0]
            where_to_delete = np.where(melody_candidates == bass_index)[0]
            melody_candidates = np.delete(melody_candidates,where_to_delete)
        if len(melody_candidates) > 1:
            predicted_melody[np.argmin(features.iloc[melody_candidates].poly_rate)] = True
            logger.debug(f'use rules to find melody track')
        if len(melody_candidates) == 1:
            predicted_melody[melody_candidates] = True
            logger.debug(f'use rules to find melody track')



    if np.sum(predicted_chord) == 0:
        chord_candicate = np.intersect1d(np.where(features['mean_dur_nor'] > 0.8),
                                         np.where(features['poly_rate'] > 0.9))
        if len(chord_candicate) > 0:
            logger.debug(f'use rules to find chord track')

            if len(chord_candicate) > 1:
                predicted_chord[(np.argmax(features.loc[chord_candicate, 'dur']))] = True
            else:
                predicted_chord[(features.index[chord_candicate[0]])] = True

    predicted_melody[predicted_drum] = False
    predicted_bass[predicted_drum] = False
    predicted_chord[predicted_drum] = False

    predicted_melody[predicted_bass] = False
    predicted_melody[predicted_chord] = False

    predicted_bass[predicted_chord] = False
    predicted_bass[predicted_melody] = False



    predicted_drum[predicted_melody] = False
    predicted_drum[predicted_bass] = False
    predicted_drum[predicted_chord] = False

    features['is_melody'] = list(map(check_melody, features['trk_names']))
    features['is_bass'] = list(map(check_bass, features['trk_names']))
    features['is_chord'] = list(map(check_chord, features['trk_names']))
    features['is_drum'] = list(map(check_drum, features['trk_names']))

    predicted_melody[features['is_drum']] = False
    predicted_bass[features['is_drum']] = False
    predicted_chord[features['is_drum']] = False


    features['melody_predict'] = predicted_melody
    features['bass_predict'] = predicted_bass
    features['chord_predict'] = predicted_chord
    features['drum_predict'] = predicted_drum
    return features


def predict(all_names, input_folder, output_folder,required_tracks,
            melody_model, bass_model, chord_model, drum_model):

    all_file_prog = {}

    for file_name in all_names:
        # logger.info(f'file name is {file_name}')
        logger.debug(f'the file is {file_name}')

        try:

            features, pm = cal_file_features(file_name)

            if pm is None:
                continue
            features = add_labels(features)

            remove_file_duplicate_tracks(features, pm)
            # logger.info(features.shape)
            features = predict_labels(features, melody_model, bass_model, chord_model,drum_model)
            # logger.info(features.shape)

            progs = []

            melody_tracks = np.count_nonzero(features.is_melody == True)

            bass_tracks = np.count_nonzero(features.is_bass == True)

            chord_tracks = np.count_nonzero(features.is_chord == True)

            drum_tracks = np.count_nonzero(features.is_drum == True)


            predicted_melody_tracks = np.count_nonzero(features.melody_predict == True)

            predicted_bass_tracks = np.count_nonzero(features.bass_predict == True)

            predicted_chord_tracks = np.count_nonzero(features.chord_predict == True)

            predicted_drum_tracks = np.count_nonzero(features.drum_predict == True)

            # if features.shape[0] < 2:
            #     logger.info(f'track number is less than 2, skip {file_name}')
            #     continue

            temp_index = []


            if melody_tracks > 0:
                temp_index.append(features.index[np.where(features.is_melody == True)][0])

            elif predicted_melody_tracks > 0:
                predicted_melody_indices = features.index[np.where(features.melody_predict == True)]

                if len(predicted_melody_indices) > 1:
                    temp_index.append(predicted_melody_indices[np.argmax(features.loc[predicted_melody_indices, 'dur'].values)])
                else:
                    temp_index.append(predicted_melody_indices[0])
            else:
                if 'melody' in required_tracks:
                    logger.info(f'no melody, skip {file_name}')
                    continue
                else:
                    temp_index.append(-1)
                    logger.debug(f'no melody track')

            # logger.info(temp_index)

            if temp_index[0] != -1:
                progs.append(features.loc[temp_index[0], 'trk_prog'])
            else:
                progs.append(-1)



            if bass_tracks > 0:
                temp_index.append(features.index[np.where(features.is_bass == True)][0])
            elif predicted_bass_tracks > 0:
                predicted_bass_indices = features.index[np.where(features.bass_predict == True)]
                if len(predicted_bass_indices) > 1:
                    temp_index.append(predicted_bass_indices[np.argmax(features.loc[predicted_bass_indices, 'dur'].values)])
                else:
                    temp_index.append(predicted_bass_indices[0])
            else:
                if 'bass' in required_tracks:
                    logger.info(f'no bass, skip {file_name}')
                    continue
                else:
                    logger.debug('no bass')
                    temp_index.append(-2)


            if temp_index[1] != -2:
                progs.append(features.loc[temp_index[1], 'trk_prog'])
            else:
                progs.append(-2)

            # logger.info(temp_index)


            if chord_tracks > 0:
                temp_index.append(features.index[np.where(features.is_chord == True)][0])
            elif predicted_chord_tracks > 0:
                predicted_chord_indices = features.index[np.where(features.chord_predict == True)]
                if len(predicted_chord_indices) > 1:
                    temp_index.append(predicted_chord_indices[np.argmax(features.loc[predicted_chord_indices, 'dur'].values)])
                else:
                    temp_index.append(predicted_chord_indices[0])
            else:
                if 'chord' in required_tracks:
                    logger.info(f'no chord, skip {file_name}')
                    continue
                else:
                    logger.debug('no chord')
                    temp_index.append(-3)

            # logger.info(temp_index)
            if temp_index[2] != -3:
                progs.append(features.loc[temp_index[2], 'trk_prog'])
            else:
                progs.append(-3)

            drum_exist = True
            if drum_tracks > 0:
                temp_index.append(features.index[np.where(features.is_drum == True)][0])
            elif predicted_drum_tracks > 0:
                predicted_drum_indices = features.index[np.where(features.drum_predict == True)]
                if len(predicted_drum_indices) > 1:

                    temp_index.append(predicted_drum_indices[np.argmax(features.loc[predicted_drum_indices, 'dur'].values)])
                else:
                    temp_index.append(predicted_drum_indices[0])
            else:

                if 'drum' in required_tracks:
                    logger.info(f'no drum, skip {file_name}')
                    continue
                else:
                    logger.debug('no drum')
                    drum_exist = False
                    temp_index.append(-5)

            if temp_index[-1] != -5:
                progs.append(0)
            else:
                progs.append(-5)


            accompaniment_track = False

            dur_sort_indices = features.dur[features.dur > 0.5].iloc[
                np.argsort(features.dur[features.dur > 0.5])].index.values


            for index in dur_sort_indices[5::-1]:
                if index not in temp_index:
                    if features.loc[index, 'trk_prog'] not in progs:
                        if features.loc[index, 'trk_prog'] == 0:
                            continue
                        temp_index.insert(-1,index)
                        accompaniment_track = True
                        break

            if 'accompaniment' in required_tracks and not accompaniment_track:
                logger.info(f'no accompaniment, skip {file_name}')
                continue
            elif not accompaniment_track:
                logger.debug('no accompaniment')
                temp_index.insert(-1,-4)
            else:
                pass

            if temp_index[-2] != -4:
                progs.insert(-1,features.loc[temp_index[-2], 'trk_prog'])

            else:
                progs.insert(-1,-4)

            # if np.sum(np.array(temp_index) >= 0) - int(drum_exist) < 2:
            #     logger.info(f'result track number is 1, skip this {file_name}')
            #     continue


            result_program = {}
            if -1 not in temp_index:
                result_program['melody'] = int(progs[0])

            if -2 not in temp_index:
                result_program['bass'] = int(progs[1])

            if -3 not in temp_index:
                result_program['chord'] = int(progs[2])

            if -4 not in temp_index:
                result_program['accompaniment'] = int(progs[-2])

            if -5 not in temp_index:
                result_program['drum'] = int(progs[-1])






            # logger.info(temp_index)
            # logger.info(progs)
            # logger.info(len(pm.instruments))
            if drum_exist:
                pm.instruments[temp_index[-1]].is_drum = True
                pm.instruments[temp_index[-1]].program = 0
            pm_new = deepcopy(pm)
            pm_new.instruments = []
            for i in temp_index:
                if i >= 0:
                    pm_new.instruments.append(deepcopy(pm.instruments[i]))


            if input_folder[-1] != '/':
                input_folder += '/'
            name_with_sub_folder = file_name.replace(input_folder,"")
            # original_names.append(file_name)
            # logger.info(file_name)
            # logger.info(len(pm.instruments))
            output_name = os.path.join(output_folder,name_with_sub_folder)
            new_output_folder = os.path.dirname(output_name)

            if not os.path.exists(new_output_folder):
                os.makedirs(new_output_folder)
            pm_new.write(output_name)
            all_file_prog[output_name] = result_program


        except Exception as e:
            logger.warning(e)

    return all_file_prog


def get_args(default='.'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default=default, type=str,
                        help="MIDI file input folder")
    parser.add_argument('-f', '--file_name', default='', type=str,
                        help="input MIDI file name")
    parser.add_argument('-o', '--output_folder',default=default,type=str,
                        help="MIDI file output folder")
    parser.add_argument('-t', '--required_tracks', default='melody',type=str,
                        help="output file criteria, a list of name for output tracks,"
                             "the list can be 'melody','bass','chord',"
                             "'accomaniment','drum'")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    running_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    melody_model = pickle.load(open(running_dir + '/melody_model','rb'))
    bass_model = pickle.load(open(running_dir+ '/bass_model','rb'))
    chord_model = pickle.load(open(running_dir+ '/chord_model','rb'))
    drum_model = pickle.load(open(running_dir + '/drum_model', 'rb'))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder,exist_ok=True)

    logger = logging.getLogger(__name__)

    logger.handlers = []

    output_json_name = os.path.join(args.output_folder, "files_result.json")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    logfile = args.output_folder + '/track_separate.log'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)

    if len(args.file_name) > 0:
        all_names = [args.file_name]
        args.input_folder = os.path.dirname(args.file_name)
    else:
        all_names = walk(args.input_folder)

    total_file_len = len(all_names)
    logger.info(f'total file {total_file_len}')
    all_file_prog = predict(all_names, args.input_folder,
                            args.output_folder,
                            args.required_tracks,
                            melody_model, bass_model, chord_model, drum_model)

    with open(os.path.join(args.output_folder,'program_result.json'), 'w') as fp:
        json.dump(all_file_prog, fp, ensure_ascii=False)
    result_file_len = len(all_file_prog.keys())
    logger.info(f'result file {result_file_len}')
    logger.info(f'ratio = {result_file_len / total_file_len}')
