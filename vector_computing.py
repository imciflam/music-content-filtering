import os
import glob
import pandas as pd
import get_preview as gp
import track_preparation as tp
import temp_audio_removal as tar
from tqdm import tqdm
from scipy.spatial import distance
import numpy as np
import time

mock_input = {'fname': ['x'],
              'Electronic': [0.379083067],
              'Experimental': [0.25455597],
              'Folk': [0.002999391],
              'Hip-Hop': [0.024724836],
              'Instrumental': [0.024724836],
              'International': [0.00712133],
              'Pop': [0.047015246],
              'Rock': [0.284489036],
              }

mock_input_df = pd.DataFrame(data=mock_input)
dataset_df = pd.read_csv('conv_results.csv')

distance_for_tracks = {}
for index in dataset_df.index:
    distance_for_tracks[dataset_df.iloc[index, 0]] = (distance.cosine(
        mock_input_df.iloc[0, 1:9], dataset_df.iloc[index, 1:9]))


sorted_distance_dictionary = sorted(
    distance_for_tracks.items(), key=lambda x: x[1])
top_five_items = take(5, sorted_distance_dictionary.items())
print(top_five_items)
# closest_index = distance_for_tracks.index(min(distance_for_tracks))
# closest_track_name = dataset_df.iloc[closest_index, 0]
