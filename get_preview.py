import json
import requests
import json
import re
import os


def top_tracks_information(items):
    for item in items:
        track_name = item['track']
        artist_name = item['artist']
        if not os.path.exists('current_track'):
            os.makedirs('current_track')
        if "preview_url" in item and item['preview_url'] != None:
            audio_url = item['preview_url']
            doc = requests.get(item['preview_url'])
            with open("current_track/" + item['artist'] + "_" + re.sub("[:*<>|?]", " ", item['track'])+'.mp3', 'wb') as f:
                f.write(doc.content)
