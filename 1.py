
import os
import glob


files = glob.glob('/converted_track/')
for f in files:
    print(f)
    os.remove(f)
