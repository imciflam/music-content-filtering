import glob
import os
import os.path


def final_audio_cleaning():
    standart_dir = os.getcwd()
    converted = standart_dir + "/converted_track/"
    current = standart_dir + "/current_track/"
    types = ("*.mp3", "*.wav")
    filelist_converted = glob.glob(os.path.join(converted, "*.wav"))
    for f in filelist_converted:
        os.remove(f)
    filelist_current = glob.glob(os.path.join(current, "*.mp3"))
    for f in filelist_current:
        os.remove(f)
