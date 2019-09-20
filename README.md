# Midi Miner
Python MIDI track classifier and tonal tension calculation based on spiral array theory
## Usage


1. **Total tension and chord calculator**. It will output three tension metrics for the midi file and chord names for each half note. The tension metrics are based on the spiral array theory proposed in [1]. It includes cloud diameter, cloud momentum and tensile strain, which correspond with \_diameter, \_centroid_diff and \_total suffix in the output files. The tension calculation window could be set by parameter -w. -w 1 will set that to default value half note. -w 2 will double that window and each bar will output one tension value.<br/> **Example:**<br/>tension_calculate.py -i _input_folder_ -o _output_folder_ -w 1<br/>
This will run tension_calculate.py on all the file in the _input_folder_ and output the result in 
_output_folder_. -w 1 is to calcualte tension for every half note.
**Example:**<br/>tension_calculate.py -i _input_folder_ -o _output_folder_ -f abc.mid -w 2<br/>
This will run tension_calculate.py on the file abc.mid in the _input_folder_ and output the result in 
_output_folder_. -w 2 is to calculate tension for every bar. 

In the example folder, 
tension_calculate.py -i input/ -o output/ -f abc.mid -w 2
will generate the tension files for abc.mid in the output folder.

files_result.json records the file key and potential key changing bar position. \_chord_name file is the chord name for every half note. The output of tension, chord, chord name is in pickle format. Below is the figure of tensile strain of old.mid for every bar.
![Tensile strain of old.mid for every bar](example/new_total.png)

2. **MIDI track separator**. Based on random forest classifier, it can find the melody, bass and harmony tracks in the MIDI file and output a new MIDI file with such tracks. If it cannot find bass or harmony tracks, it will only output a MIDI file if it detects a melody track.<br/>  **Example:** <br/> track_separate.py -i _input_folder_ -o _output_folder_<br/>
input_folder contains the original midi files, and output_folder is the destination for the new MIDI file.

In the example folder, new.mid is the result by running track_separate.py on the old.mid file. 


[1] E. Chew. Mathematical and computational modeling of tonality. AMC, 10:12, 2014.
