import urllib.request
import json
import urllib.parse
import requests
import json


def preview_download():
    track_url = 'https://api.spotify.com/v1/tracks/3rCtueI7qBN2kZBZnXuk5K'
    bearer = 'BQDnjDy9PYzdUmGpjqq4f16e_HKHEAe0F_VdRaNddCsVPy2SDyoAf1tTuTZbRevpfU9uXpizHVFSir9pkPqzpSXOdYQPD6QJBkmKl2i_oRgHGPKCoILsO5FVC77kJRfqqooZhazWnb1FRs781rbI7MRmu8AJMevcIRQE04I'
    headers = {'Authorization': 'Bearer ' + bearer}
    response = requests.get(track_url, headers=headers)
    if response.status_code == 200:
        print('Received track metadata successfully, code ' +
              str(response.status_code))
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
        print('Request failed, code ' + str(response.status_code))
        return 0
