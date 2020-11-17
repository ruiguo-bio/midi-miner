# Midi Miner
Python MIDI track classifier and tonal tension calculation based on spiral array theory
## Usage
Please refer to the [example notebook](example.ipynb) file for detailed examples.

1. **Total tension calculation**. It will output three tension metrics for the midi file. The tension metrics are based on the spiral array theory proposed in [1], which includes cloud diameter, cloud momentum and tensile strain. The default tension calculation window size is one bar. The tension calculation length window could be set by parameter -w. -w 1 set the window to 1 beat, -w 2 means 2 bar, and -1 is default for a downbeat (1 bar). <br/> **Example:**<br/>tension_calculation.py -i _input_folder_ -o _output_folder_ -k True<br/>
This will run tension_calculation.py on all the file in the _input_folder_ and output the result in 
_output_folder_. -k True means it tries to find key change. Default is not, which is not detecting key change.
It will try to find one key change in the song which is usually in pop songs, but not the classical songs.

The vertical step in the spiral array theory can be changed by -v parameter, which should be between sqrt(2/15) and sqrt(0.2). The current implementation set it to 0.4.



files_result.json records the file key and potential key changing time and bar position. The output of three tension measures are in pickle format. 

2. **MIDI track separator**. Based on random forest classifier, it can find the melody, bass, chord, and drum tracks in the MIDI file and output a new MIDI file with such tracks, including one potential accompaniment track. Use -t to specify the required tracks, e.g -t "melody bass" will omit the files without both a melody and a bass tracks detected. The default is 'melody'. <br/>  **Example:** <br/> track_separate.py -i _input_folder_ -o _output_folder_  -t "melody bass drum" <br/>
input_folder contains the original midi files, and output_folder is the destination for the new MIDI file. Use -f to select one file.


[1] E. Chew. Mathematical and computational modeling of tonality. AMC, 10:12, 2014.

[2] Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching"

