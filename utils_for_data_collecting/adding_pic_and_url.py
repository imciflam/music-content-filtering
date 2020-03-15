import json
import requests
import json
import re
import pandas as pd
import spotipy
from spotipy import oauth2
from spotipy.oauth2 import SpotifyClientCredentials


SPOTIPY_CLIENT_ID = '032bb2c730e645968318b1811d084943'
SPOTIPY_CLIENT_SECRET = '5892fdf2a41948e1ae73078e4cecb6f2'


client_credentials_manager = SpotifyClientCredentials(
    SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
sp_limited = spotipy.Spotify(
    client_credentials_manager=client_credentials_manager)


def add_data():
    # 1. foreach element in dataset - search for a track on spotify
    # 2. append pic and url to dataset
    dataset_df = pd.read_csv('../spotify_preview_dataset.csv')
    pics = []
    links = []
    for index in dataset_df.index:
        element = dataset_df.iloc[index, 0]
        element = element.replace('_', ' ')
        element = element.replace('.wav', '')
        searchItems = sp_limited.search(q=element[0], type='track')
        if searchItems["tracks"]['items'] != []:
            track = searchItems["tracks"]["items"][0]
            if "preview_url" in track and track['preview_url'] != None:
                links.append(track['preview_url']) 
            else:
                links.append("")
            if track['album']['images'][0]['url']!= None:
                pics.append(track['album']['images'][0]['url'])
            else:
                print("no pic")
                pics.append("https://sun7-8.userapi.com/mfdJIv5AQYX3tkWyzCtqMpW9LRMkBrkw69sDrg/V215K1wSxFE.jpg")

        else:
            print("no track")

    print(len(links))
    print(len(pics))
    dataset_df['preview_url'] = links
    dataset_df['picture'] = pics
    
    dataset_df.to_csv('spotify_preview_dataset.csv', index=False)


add_data()
