import json
import requests
import json
import re
import pandas as pd


def add_data():
    # 1. foreach element in dataset - search for a track on spotify
    # 2. append pic and url to dataset

    dataset_df = pd.read_csv('./spotify_preview_dataset.csv')
    pics = []
    links = []
    bearer = 'BQBCDa2C3YNDKizW10X69s9l_Exe3qnU5vhZnM9s8illHgW_ecj0w3hkTpcJ49ag4E-D_lqjebLwWalV6IYAUwgNLOkPALvZpKR2-eBbBuge2uJtlQgznAQhjZs6iYo88bN8BXBsOL9JFJKYIaiIJ7qYDvU6uego7IncwdE'
    headers = {'Authorization': 'Bearer ' + bearer}
    for index in dataset_df.index:
        element = dataset_df.iloc[index, 0]
        element = element.replace('_', ' ')
        element = element.replace('.wav', '')
        request_link = "https://api.spotify.com/v1/search?q=" + \
            element + "&type=track&limit=1"
        response = requests.get(request_link, headers=headers)
        parsed = json.loads(response.text)
        if parsed['tracks']['total'] != 0:
            if parsed['tracks']['items'][0]['album']['images'][0]['url'] != None:
                pics.append(parsed['tracks']['items'][0]
                            ['album']['images'][0]['url'])
            else:
                print("no images")
                pics.append(
                    "https://sun7-8.userapi.com/mfdJIv5AQYX3tkWyzCtqMpW9LRMkBrkw69sDrg/V215K1wSxFE.jpg")
            if parsed['tracks']['items'][0]['preview_url'] != None:
                links.append(parsed['tracks']['items'][0]['preview_url'])
            else:
                print("no url")
                links.append("")
        # pics.append("https://sun7-8.userapi.com/mfdJIv5AQYX3tkWyzCtqMpW9LRMkBrkw69sDrg/V215K1wSxFE.jpg")
        else:
            print(element)
    dataset_df['preview_url'] = links
    dataset_df['picture'] = pics

    dataset_df.to_csv('spotify_preview_dataset.csv', index=False)


add_data()
