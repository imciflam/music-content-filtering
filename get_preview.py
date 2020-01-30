import urllib.request
import json
import urllib.parse
import requests
import json


def top_tracks_information():
    top_url = 'https://api.spotify.com/v1/me/top/tracks?time_range=short_term&limit=5'
    bearer = 'BQBeLytGf7lRo10s5ROV_exAKemkn_VIttovccpmdEhaqROFiGa3gdH5EVS48h_JqWrLGGeP8eJ8-6NIBIFQ4uaIvdS6h_Cf3AFLWJ1cOrqNqxCJmav-hGFyQRzus618qnS_kChK7m0ApjShROipDjezjtAanaphxKjIhx8'
    headers = {'Authorization': 'Bearer ' + bearer}
    response = requests.get(top_url, headers=headers)
    if response.status_code == 200:
        print('Received track metadata successfully, code ' +
              str(response.status_code))
        parsed = json.loads(response.text)
        items = parsed['items']
        for item in items:
            track_name = item['name']
            artist_name = item['artists'][0]['name']
            if "preview_url" in item and item['preview_url'] != None:
                audio_url = item['preview_url']
                get_audio = urllib.request.urlretrieve(
                    audio_url,  "current_track/" + artist_name+'_'+track_name+'.mp3')
            else:
                print('Link for audio '+track_name+' wasn\'t found')
        return 1
    else:
        print('Request failed, code ' + str(response.status_code))
        return 0
