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

mock_input_df = pd.read_csv("conv_results.csv")
dataset_df = pd.read_csv('conv_results_for_dataset.csv')


def cosine_distance_calculation():
    distance_for_tracks = {}
    output_data = []
    for input_index in mock_input_df.index:
        for index in dataset_df.index:
            distance_for_tracks[dataset_df.iloc[index, 0]] = (distance.cosine(
                mock_input_df.iloc[input_index, 1:9], dataset_df.iloc[index, 1:9]))
        sorted_distance_dictionary = sorted(
            distance_for_tracks.items(), key=lambda x: x[1])
        top_five_items = sorted_distance_dictionary[:5]
        output_data.append(top_five_items)
    return output_data


def jensen_shannon_distance_calculation():
    distance_for_tracks_jensenshannon = {}
    output_data = []
    for input_index in mock_input_df.index:
        for index in dataset_df.index:
            distance_for_tracks_jensenshannon[dataset_df.iloc[index, 0]] = (distance.cosine(
                list(mock_input_df.iloc[input_index, 1:9].to_numpy()), list(dataset_df.iloc[index, 1:9].to_numpy())))
        sorted_distance_dictionary_jensenshannon = sorted(
            distance_for_tracks_jensenshannon.items(), key=lambda x: x[1])
        top_five_items_jensenshannon = sorted_distance_dictionary_jensenshannon[:5]
        output_data.append(top_five_items_jensenshannon)
    return output_data


print(cosine_distance_calculation())
