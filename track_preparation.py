import os
import glob
from pydub import AudioSegment

standart_dir = os.getcwd()
audio_dir = os.path.join(os.getcwd(), 'current_track')
extension = ('*.mp3')
os.chdir(audio_dir)
for audio in glob.glob(extension):
    wav_filename = os.path.splitext(os.path.basename(audio))[0] + '.wav'
    AudioSegment.from_file(audio).export(
        standart_dir + "/converted_track/" + wav_filename, format='wav')

os.chdir(standart_dir)
