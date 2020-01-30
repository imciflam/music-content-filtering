import urllib.request
import json
import urllib.parse
import requests
import json


def top_tracks_information():
    top_url = 'https://api.spotify.com/v1/me/top/tracks?time_range=short_term&limit=5'
    bearer = 'BQB_X0Tof2mLnSwmyuB4sqM2LlsAg8fBdMk_6yLNIjlGXq3jpL4PLxLpZc9b81ZjFdPXZ88dFf2MoMXQlB_IjA4eqlbQzZJtRgHamD4WDkdAfJqm8CZwTOB6sVQ28dVJ8DXlzErVkvJdJRoiN2UZRLGAqk7ptk4KRVJt2OE'
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
            print(track_name)
            print(artist_name)
            if "preview_url" in item and item['preview_url'] != None:
                audio_url = item['preview_url']
                get_audio = urllib.request.urlretrieve(
                    audio_url,  "current_track/" + artist_name+'_'+track_name+'.mp3')
            else:
                print('Link for audio wasn\'t found')
        return 1
    else:
        print('Request failed, code ' + str(response.status_code))
        return 0


top_tracks_information()
