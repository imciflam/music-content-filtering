import urllib.request
import json
import urllib.parse
import requests
import json


def preview_download():
    track_url = 'https://api.spotify.com/v1/tracks/3rCtueI7qBN2kZBZnXuk5K'
    bearer = 'BQB2NCjpIEH0jkn9hU1JoLoa6BjIvvIW3dBbezPIVsMyphDI2B7GvdvvcRp5S-No5toSbKpBZdomL20bE_NNEfwgC5BEojdXPK6YRnvcJaU8V9NTWEnoClLdDqRxLCAeKaueYt3xGP1rZZnYIQSx8Yrwv78XmDtjfCuJqNs'
    headers = {'Authorization': 'Bearer ' + bearer}
    response = requests.get(track_url, headers=headers)
    if response.status_code == 200:
        print('Received track metadata successfully, code ' + response.status_code)
        parsed = json.loads(response.text)
        track_name = parsed['name']
        artist_name = parsed['artists'][0]['name']
        if "preview_url" in parsed:
            audio_url = parsed['preview_url']
            get_audio = urllib.request.urlretrieve(
                audio_url,  "current_track/" + artist_name+'_'+track_name+'.mp3')
            return 1
        else:
            print('Link for audio wasn\'t found')
    else:
        print('Request failed, code '+response.status_code)
        return 0
