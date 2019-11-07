import _pickle as pickle

import os
import pretty_midi
import sys
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import math
import json


import argparse

from itertools import combinations


octave = 12

C = tuple((0,4,7))
Cm = tuple((0,3,7))
Csus4 = tuple((0,5,7))
Csus6 = tuple((0,7,9))
Dm = tuple((2,5,9))
D = tuple((2,6,9))
Dsus4 = tuple((2,7,9))
Em = tuple((4,7,11))
E = tuple((4,8,11))
F = tuple((0,5,9))
Fm = tuple((0,5,8))
G = tuple((2,7,11))
Gm = tuple((2,7,10))
Gsus4 = tuple((0,2,7))
Am = tuple((0,4,9))
Asus7 = tuple((4,7,9))
A = tuple((1,4,9))
H = tuple((3,6,11))
Hverm = tuple((2,5,11))
Hm = tuple((2,6,11))
B = tuple((2,5,10))
Es = tuple((3,7,10))
As = tuple((0,3,8))
Des = tuple((1,5,8))
Fis = tuple((1,6,10))

chord_name_to_index = {'C':C,'Cm':Cm,'Csus4':Csus4,
                        'Csus6':Csus6,'Dm':Dm,'D':D,
                        'Dsus4':Dsus4,'Em':Em,'E':E,
                        'F':F,'Fm':Fm,'G':G,'Gm':Gm,
                   'Gsus4':Gsus4,'Am':Am,'Asus7':Asus7,
                   'A':A,'H':H,'Hverm':Hverm,'Hm':Hm,'B':B,
                   'Es':Es,'As':As,
                   'Des':Des,'Fis':Fis}


pitch_index_to_sharp_names = ['C','C#','D','#D','E','F','#F','G',
                              '#G','A','#A','B']


pitch_index_to_flat_names = ['C','Db','D','Eb','E','F','Gb','G',
                             'Ab','A','Bb','B']

sharp_index = [0,2,4,7,9,11]
flat_index = [1,3,5,6,8,10]

note_index_to_pitch_index = [[0, 7, 2, 9, 4, -1, 6,  1, 8, 3, 10, 5],
                             [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5],
                             [0, 7, 2, 9, 4, -1, 6,  1, 8, 3, 10, 5],
                             [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5],
                             [0, 7, 2, 9, 4, -1, 6,  1, 8, 3, 10, 5],
                             [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5],
                             [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, -7],
                             [0, 7, 2, 9, 4, -1, 6,  1, 8, 3, 10, 5],
                             [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5],
                             [0, 7, 2, 9, 4, -1, 6,  1, 8, 3, 10, 5],
                             [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5],
                             [0, 7, 2, 9, 4, -1, 6,  1, 8, 3, 10, 5]]


index_to_incomplete_chord_name = {}

chord_name_to_incomplete_index = {
'C':tuple((0,4)),
'Cm':tuple((0,3)),
'Dm' : tuple((2, 5)),
'D' : tuple((2, 6)),
'Em' : tuple((4, 7)),
'E' : tuple((4, 8)),
'F' : tuple((5, 9)),
'Fm' : tuple((5, 8)),
'G' : tuple((7, 11)),
'Gm' : tuple((7, 10)),
'Am' : tuple((0,9)),
'A' : tuple((1,9)),
'H' : tuple((3,11)),
'Hm' : tuple((2, 11)),
}

incomplete_index_to_chord_name={}

for key,value in chord_name_to_incomplete_index.items():
    incomplete_index_to_chord_name[value] = key

index_to_chord_name = {}
for key,value in chord_name_to_index.items():
    index_to_chord_name[value] = key

weight = np.array([0.536, 0.274, 0.19])

verticalStep = math.sqrt(2.0/15.0)
radius = 1.0



def diameter(piano_roll, shift, key_changed_bar, shift_new, window_size=4):

    diameters = []
    for i in range(0, piano_roll.shape[1]):
        indices = []
        for index, j in enumerate(piano_roll[:,i]):
            if j > 0:
                if i / 2 > key_changed_bar:
                    if key_changed_bar != -1:
                        indices.append(note_index_to_pitch_index[shift_new][index % 12])
                    else:
                        indices.append(note_index_to_pitch_index[shift][index % 12])
                else:
                    indices.append(note_index_to_pitch_index[shift][index % 12])

        diameters.append(cal_diameter(indices))

    merged_diameter = []
    for time_step in range(0, len(diameters) - window_size, window_size):
        merged_diameter.append(np.mean(diameters[time_step:time_step + window_size], axis=0))
    if time_step != len(diameters) - 1:
        merged_diameter.append(np.mean(diameters[time_step:], axis=0))

    return merged_diameter


def cal_diameter(pitches):
    if len(pitches) == 0:
        return 0
    diameter = 0
    pitch_pairs = itertools.combinations(pitches,2)
    for pitch_pair in pitch_pairs:
        distance = np.linalg.norm(pitch_index_to_position(pitch_pair[0]) - pitch_index_to_position(pitch_pair[1]))
        if distance > diameter:
            diameter = distance
    return diameter


def piano_roll_to_ce(piano_roll,shift):

    pitch_index = []
    for i in range(0, piano_roll.shape[1]):
        indices = []
        for index, j in enumerate(piano_roll[:,i]):
            if j > 0:
                indices.append(note_index_to_pitch_index[shift][index % 12])

        pitch_index.append(indices)

    ce_pos = ce_sum(pitch_index)
    return ce_pos


def notes_to_ce(notes,shift):
    indices = []

    for index, j in enumerate(notes):
        if j > 0:
            indices.append(note_index_to_pitch_index[shift][index % 12])

    total = np.zeros(3)
    count = 0
    for index in indices:
        total += pitch_index_to_position(index)
        count += 1
    if count == 0:
        return total
    return total / count


def pitch_index_to_position(pitch_index):

    c = pitch_index - (4 * (pitch_index // 4))

    pos = np.array([0.0, 0.0, 0.0])

    if c == 0:
        pos[1] = radius
    if c == 1:
        pos[0] = radius
    if c == 2:
        pos[1] = -1*radius
    if c == 3:
        pos[0] = -1*radius

    pos[2] = pitch_index * verticalStep
    return np.array(pos)


def ce_sum(indices, start=None, end=None):
    if not start:
        start = 0
    if not end:
        end = len(indices)

    indices = indices[start:end]
    total = np.zeros(3)
    count = 0
    for timestep, data in enumerate(indices):
        for pitch in data:
            total += pitch_index_to_position(pitch)
            count += 1
    return total/count


def major_triad_position(root_index):
    root_pos = pitch_index_to_position(root_index)

    fifth_index = root_index + 1
    third_index = root_index + 4

    fifth_pos = pitch_index_to_position(fifth_index)
    third_pos = pitch_index_to_position(third_index)

    centre_pos = weight[0] * root_pos + weight[1] * fifth_pos + weight[2] * third_pos
    return centre_pos


def minor_triad_position(root_index):
    root_pos = pitch_index_to_position(root_index)

    fifth_index = root_index + 1
    third_index = root_index - 3

    fifth_pos = pitch_index_to_position(fifth_index)
    third_pos = pitch_index_to_position(third_index)

    centre_pos = weight[0] * root_pos + weight[1] * fifth_pos + weight[2] * third_pos

    return centre_pos


def major_key_position(key_index):
    root_triad_pos = major_triad_position(key_index)
    fifth_index = key_index + 1

    fourth_index = key_index - 1

    fifth_triad_pos = major_triad_position(fifth_index)
    fourth_triad_pos = major_triad_position(fourth_index)

    key_pos = weight[0] * root_triad_pos + weight[1] * fifth_triad_pos + weight[2] * fourth_triad_pos

    return key_pos


def minor_key_position(key_index):

    root_triad_pos = minor_triad_position(key_index)
    fifth_index = key_index + 1
    fourth_index = key_index - 1
    minor_fourth_triad_pos = minor_triad_position(fourth_index)
    major_fifth_triad_pos = major_triad_position(fifth_index)

    key_pos = weight[0] * root_triad_pos + weight[1] * major_fifth_triad_pos + weight[2] * minor_fourth_triad_pos

    return key_pos


def cal_key(piano_roll, key_index,key_given=False,is_minor=False):
    end = int(piano_roll.shape[1] * 0.5)
    ce = piano_roll_to_ce(piano_roll[:,:end],key_index)

    return key_search(ce, key_index,key_given=key_given,is_minor=is_minor)


def key_search(ce, shift,key_given=False,is_minor=False):

    major_key_pos = major_key_position(note_index_to_pitch_index[shift][shift])

    minor_shift = shift - 3
    if minor_shift < 0:
        minor_shift += 12

    minor_key_pos = minor_key_position(note_index_to_pitch_index[shift][minor_shift])
    if is_minor:
        result = f'minor {pitch_index_to_flat_names[minor_shift]}'
        # print(result)
        return minor_key_pos,result
    diff_major = np.linalg.norm(ce - major_key_pos)

    diff_minor = np.linalg.norm(ce - minor_key_pos)

    if shift in sharp_index:
        if key_given:
            result = f'major {pitch_index_to_sharp_names[shift]}'
            # print(result)
            return major_key_pos,result
        if diff_major < diff_minor:
            result = f'major {pitch_index_to_sharp_names[shift]}'
            # print(result)
            return major_key_pos,result
        else:
            result = f'minor {pitch_index_to_sharp_names[minor_shift]}'
            # print(result)
            return minor_key_pos,result
    else:
        if key_given:
            result = f'major {pitch_index_to_flat_names[shift]}'
            # print(result)
            return major_key_pos,result

        if diff_major < diff_minor:
            result = f'major {pitch_index_to_flat_names[shift]}'
            # print(result)
            return major_key_pos,result
        else:
            result = f'minor {pitch_index_to_flat_names[minor_shift]}'
            # print(result)
            return minor_key_pos,result


def pianoroll_to_pitch(pianoroll):
    pitch_roll = np.zeros((12, pianoroll.shape[1]))
    for i in range(0, pianoroll.shape[0]-12+1, 12):
        pitch_roll = np.add(pitch_roll, pianoroll[i:i+octave])
    return np.transpose(pitch_roll)


def note_to_index(pianoroll):
    note_ind = np.zeros((128,pianoroll.shape[1]))
    for i in range(0, pianoroll.shape[1]):
        step = []
        for j, note in enumerate(pianoroll[:, i]):
            if note != 0:
                step.append(j)
        if len(step) > 0:
            note_ind[step[-1],i] = 1
    return np.transpose(note_ind)


def merge_tension(metric,window_size=1):
    if window_size == 1:
        return metric, beats
    new_metric = []
    if len(metric.shape) > 1:
        for i in range(0, len(metric), window_size):
            new_metric.append(np.mean(metric[i:i + window_size],axis=0))
    else:
        for i in range(0, len(metric), window_size):
            new_metric.append(np.mean(metric[i:i + window_size]))
    return new_metric


def cal_tension(file_name, pm, beats, output_folder, window_size=1, key_index=None, is_minor=False):
    try:

        # pm = pretty_midi.PrettyMIDI(file_name)
        # pm = remove_drum_track(pm)
        # print(file_name)

        base_name = os.path.basename(file_name)


        if key_index is None:
            key_index, is_minor = get_key_index(pm)
            key_given = False
        else:
            key_given = True

        if key_index is None:
            return None,None

        chord_roll_eighth = pickle.load(open(os.path.join(output_folder,base_name[:-4]+'_chord_eighth'),'rb'))
        chord = pickle.load(open(os.path.join(output_folder,base_name[:-4]+'_chord'),'rb'))



        key_pos,key_name = cal_key(chord_roll_eighth, key_index, key_given,is_minor)

        centroids = get_centroid(chord, key_index,window_size=1)
        silent = np.where(np.linalg.norm(centroids, axis=-1) == 0)
        centroids = np.array(centroids)

        key_diff = centroids - key_pos
        key_diff = np.linalg.norm(key_diff, axis=-1)

        key_diff[silent] = 0

        key_change_bar, change_time = detect_key_change(key_diff, pm)

        if key_change_bar != -1:
            ## assume 4/4 beat
            key_index_new, is_minor_new = get_key_index_change(pm, change_time)
            change_key_pos, change_key_name = cal_key(chord_roll_eighth[:, :key_change_bar * 8], key_index_new, key_given,
                                                      is_minor_new)
            # print(f'new key name is {change_key_name}')
        else:
            change_key_name = ''

        diameters = diameter(chord, key_index,key_change_bar, key_index_new, window_size=1)

        diameters = np.array(diameters)



        key_diff = merge_tension(key_diff,window_size)
        diameters = merge_tension(diameters,window_size)

        new_centroid = merge_tension(centroids,window_size)

        centroid_diff = np.diff(new_centroid, axis=0)

        np.nan_to_num(centroid_diff, copy=False)

        centroid_diff = np.linalg.norm(centroid_diff, axis=-1)
        centroid_diff = np.insert(centroid_diff, 0, 0)

        total_tension = np.array(key_diff) / np.max(key_diff)
        diameters = np.array(diameters) / np.max(diameters)
        centroid_diff = np.array(centroid_diff) / np.max(centroid_diff)

        pickle.dump(total_tension, open(os.path.join(output_folder,
                                                     base_name[:-4]+'_tensile_strain'),
                                        'wb'))

        pickle.dump(diameters, open(os.path.join(output_folder,
                                                base_name[:-4] + '_diameter'),
                                   'wb'))
        pickle.dump(centroid_diff, open(os.path.join(output_folder,
                                                base_name[:-4] + '_centroid_diff'),
                                   'wb'))
        draw_tension(total_tension,os.path.join(output_folder,
                                                     base_name[:-4]+'_tensile_strain.png'))
        draw_tension(diameters, os.path.join(output_folder,
                                                 base_name[:-4] + '_diameter.png'))
        draw_tension(centroid_diff, os.path.join(output_folder,
                                             base_name[:-4] + '_centroid_diff.png'))

        return total_tension, diameters, centroid_diff, key_name,key_change_bar, change_key_name,beats[::window_size]

    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
        exception_str = 'Unexpected error in ' + file_name + ':\n', e, sys.exc_info()[0]
        print(exception_str)

def get_scales():
    # get all scales for every root note
    dia = tuple((0, 2, 4, 5, 7, 9, 11))
    diatonic_scales = []
    for i in range(0, 12):
        diatonic_scales.append(tuple(np.sort((np.array(dia) + i) % 12)))

    harm = tuple((0, 2, 4, 5, 8, 9, 11))
    harmonic_scales = []
    for i in range(0, 12):
        harmonic_scales.append(tuple(np.sort((np.array(harm) + i) % 12)))

    return diatonic_scales, harmonic_scales


def get_key_index(pm):
    end_time = pm.get_end_time() * 0.5
    new_pm = copy.deepcopy(pm)
    for instrument in new_pm.instruments:
        for i,note in enumerate(instrument.notes):
            if note.end > end_time:
                instrument.notes = instrument.notes[:i]
                break
    pitches = new_pm.get_pitch_class_histogram(use_duration=True)
    frequent = pitches.argsort(axis=0)[-7:]
    frequent.sort()
    frequent = tuple(frequent)

    diatonic_scales,harmonic_scales = get_scales()
    if frequent in diatonic_scales:
        return diatonic_scales.index(frequent),False
    if frequent in harmonic_scales:
        return harmonic_scales.index(frequent),True
    return None,None


def get_key_index_change(pm,start_time):

    new_pm = copy.deepcopy(pm)
    for instrument in new_pm.instruments:
        for i,note in enumerate(instrument.notes):
            if note.start > start_time:
                instrument.notes = instrument.notes[i:]
                break
    pitches = new_pm.get_pitch_class_histogram(use_duration=True)
    frequent = pitches.argsort(axis=0)[-7:]
    frequent.sort()
    frequent = tuple(frequent)

    diatonic_scales,harmonic_scales = get_scales()
    if frequent in diatonic_scales:
        return diatonic_scales.index(frequent),False
    if frequent in harmonic_scales:
        return harmonic_scales.index(frequent),True
    return None,None


def note_pitch(melody_track):

    pitch_sum = []
    for i in range(0, melody_track.shape[1]):
        indices = []
        for index, j in enumerate(melody_track[:, i]):
            if j > 0:
                indices.append(index-24)

        pitch_sum.append(np.mean(indices))
    return pitch_sum


def get_piano_roll(pm,beat_times):
    piano_roll = pm.get_piano_roll(times=beat_times)
    np.nan_to_num(piano_roll, copy=False)
    piano_roll = piano_roll > 0
    piano_roll = piano_roll.astype(int)

    return piano_roll

def get_centroid(piano_roll,key_index,window_size=4):
    # widow_size=4 means to merge notes in a half note window
    centroids = list()

    for time_step in range(0,piano_roll.shape[1]):
        roll = piano_roll[:, time_step]
        centroids.append(notes_to_ce(roll, key_index))
    merged_centroids = []
    for time_step in range(0,len(centroids)-window_size,window_size):
        merged_centroids.append(np.mean(centroids[time_step:time_step+window_size],axis=0))
    if time_step != len(centroids) - 1:
        merged_centroids.append(np.mean(centroids[time_step:],axis=0))
    return merged_centroids


def detect_key_change(key_diff, pm):
    ratios = []

    # 16 means 16 half notes, 32 beats
    for i in range(16, key_diff.shape[0]-16):
        if np.any(key_diff[i - 16:i + 16] == 0):
            ratios.append(0)
            continue
        else:
            previous = np.mean(key_diff[i-16:i])
            current = np.mean(key_diff[i:i+16])
            ratio = current/previous
            ratios.append(ratio)
        #print(f'the position is {i}, and ratio is {ratio}')
    ratios = np.array(ratios)
    start = int(ratios.shape[0] * 0.5)
    for i in range(start,ratios.shape[0], 2):
        if np.mean(ratios[i:i+4]) > 2:

            down_beats = pm.get_downbeats()
            beats = pm.get_beats()
            bar = np.where(beats[32 + i*2] < down_beats)[0][0]
            # print(f'the key changed after bar {bar + 2}')

            return bar + 2, down_beats[bar+2]
    return -1,-1


def draw_tension(values,file_name):

    plt.rcParams['xtick.labelsize'] = 6
    plt.figure()
    F = plt.gcf()
    F.set_size_inches(len(values)/7+5, 5)
    xtick = range(1,len(values) + 1)
    if isinstance(values,list):
        values = np.array(values)

    plt.xticks(xtick)
    minimum = np.min(values[values!=0])
    maximum = np.max(values)
    plt.ylim(minimum, maximum)
    plt.scatter(xtick, values)
    plt.savefig(file_name)
    plt.close('all')


def merge_piano_roll(piano_roll, window_size=4):
    all_notes = []
    for i in range(0, piano_roll.shape[1], window_size):
        note_in_window = np.sum(piano_roll[:, i:(i + window_size)], axis=1)
        all_notes.append(note_in_window)
    all_notes = np.array(all_notes)
    all_notes = np.transpose(all_notes)
    return all_notes



def remove_drum_track(pm):

    for instrument in pm.instruments:
        if instrument.is_drum:
            pm.instruments.remove(instrument)
    return pm


def extract_notes(file_name,output_folder):
    try:
        pm = pretty_midi.PrettyMIDI(file_name)
        pm = remove_drum_track(pm)
        # print(file_name)

        base_name = os.path.basename(file_name)
        beats = pm.get_beats()
        beats = np.append(beats, pm.get_end_time())
        eighth_beats = []
        for i in range(len(beats) - 1):
            eighth_beats.append(beats[i])
            eighth_beats.append(beats[i] + (beats[i + 1] - beats[i]) / 2)
        eighth_beats.append(beats[-1])
        eighth_beats = np.array(eighth_beats)

        piano_roll = get_piano_roll(pm, eighth_beats)

        # window_size = 4 means half note is the window length(4*eighth note)
        chord_note_merged = merge_piano_roll(piano_roll,window_size=4)

        chord_names = chord_roll_to_chord_name(chord_note_merged)

        pickle.dump(chord_note_merged, open(os.path.join(output_folder, base_name[:-4]+'_chord'), 'wb'))
        pickle.dump(chord_names, open(os.path.join(output_folder, base_name[:-4] + '_chord_name'), 'wb'))
        pickle.dump(piano_roll, open(os.path.join(output_folder, base_name[:-4] + '_chord_eighth'), 'wb'))


    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
        exception_str = 'Unexpected error in ' + file_name + ':\n', e, sys.exc_info()[0]
        print(exception_str)
        return None

    return pm,chord_names,chord_note_merged,eighth_beats

def chord_index_combination(chord_index):
    comb = combinations(chord_index, 3)
    for chord in list(comb):
        chord_tuple = tuple(np.sort(chord))
        if chord_tuple in index_to_chord_name:
            return index_to_chord_name[chord_tuple]
    return 'None'


def chord_roll_to_chord_name(chord_roll):
    pitch_roll = pianoroll_to_pitch(chord_roll)
    chord_name = []
    for chord in pitch_roll:
        if np.sum(chord > 0) > 2:
            if np.sum(chord > 0) == 3:
                if tuple(np.where(chord > 0)[0]) in index_to_chord_name:
                    chord_name.append(index_to_chord_name[tuple(np.where(chord > 0)[0])])
                else:
                    chord_name.append('None')
            else:
                chord_index = np.where(chord > 0)[0]
                index_sorted = chord_index[np.argsort(chord[chord_index])]
                chord_name.append(chord_index_combination(index_sorted[::-1]))
        else:
            if np.sum(chord > 0) == 2:
                if tuple(np.where(chord > 0)[0]) in incomplete_index_to_chord_name:
                    chord_name.append(incomplete_index_to_chord_name[tuple(np.where(chord > 0)[0])])
                else:
                    chord_name.append('None')
            else:
                chord_name.append('None')

    return chord_name


def get_args(default='.'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default=default, type=str,
                        help="MIDI file input folder")
    parser.add_argument('-f', '--file_name', default='', type=str,
                        help="input MIDI file name")
    parser.add_argument('-o', '--output_folder',default=default,type=str,
                        help="MIDI file output folder")
    parser.add_argument('-w', '--window_size', default=1, type=int,
                        help="Tension calculation window size, default 1 for half note, double this will double the window")


    return parser.parse_args()

if __name__== "__main__":
    args = get_args()
    files_result = {}
    for path, _, files in os.walk(args.input_folder):
        for name in files:

            if name.endswith('mid') or name.endswith('MID'):
                file_name = os.path.join(path,name)


                base_name = os.path.basename(file_name)
                if len(args.file_name) > 0:
                    # base_name1 = os.path.basename(args.file_name)
                    if args.file_name != file_name:
                        continue
                pm,chord_names,chord_note,beats = extract_notes(file_name, args.output_folder)
                if not pm:
                    continue
                total_tension, diameters, centroid_diff, key_name,key_change_bar,key_change_name, beats= cal_tension(file_name, pm, beats, args.output_folder, args.window_size)
                if key_name is not None:
                    files_result[base_name[:-4]] = []
                    files_result[base_name[:-4]].append(key_name)
                    files_result[base_name[:-4]].append(int(key_change_bar))
                    files_result[base_name[:-4]].append(key_change_name)
                    print(f'file name is {file_name}')
                    print(f'key name is {key_name}')
                    if key_change_bar != -1:
                        print(f'key change bar is {key_change_bar}')
                        print(f'new key name is {key_change_name}')

                else:
                    print(f'cannot find the key of song {file_name}, skip this file')

    with open(os.path.join(args.output_folder,'files_result.json'), 'w') as fp:
        json.dump(files_result, fp)



