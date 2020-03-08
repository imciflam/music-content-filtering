import json
import requests
import json
import re


def top_tracks_information():
    top_url = 'https://api.spotify.com/v1/tracks?ids=' + \
        '7t5hpYtCNs6tTLQrfqKnBE' + \
        '%2C' + '3qwVqJyXKNiPZLz9VBMd6r' + \
        '%2C' + '0ztlJeG8IRiGMgLV5SNtzj'
    bearer = 'BQB_X3W457yZiWzrhXyEdGGMW0ln-79grKYq9XJIqrnVx8qLSBJYjzQdknXA0KuDdXp0bvWdjtKfm3kFSQNDEP1T92PG-c4kUfABxyBgBebJjXNY3aaVYjoL5tXN0eeMMIJD8rh1Qad9clw2tnArT62XvApOSqnmzzoyVuM'
    headers = {'Authorization': 'Bearer ' + bearer}
    response = requests.get(top_url, headers=headers)
    if response.status_code == 200:
        print('Received track metadata successfully, code ' +
              str(response.status_code))
        parsed = json.loads(response.text)
        items = parsed['tracks']
        for item in items:
            track_name = item['name']
            artist_name = item['artists'][0]['name']
            if "preview_url" in item and item['preview_url'] != None:
                audio_url = item['preview_url']
                doc = requests.get(item['preview_url'])
                # change path
                with open("down/Instrumental/" + item['artists'][0]['name'] + "_" + re.sub("[:*<>|?]", " ", item['name'])+'.mp3', 'wb') as f:
                    f.write(doc.content)

            else:
                print('Link for audio '+track_name+' wasn\'t found')
        return 1
    else:
        print('Request failed, code ' + str(response.status_code))
        return 0


top_tracks_information()
