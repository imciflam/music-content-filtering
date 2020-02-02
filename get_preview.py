import json
import requests
import json
import re


def top_tracks_information():
    top_url = 'https://api.spotify.com/v1/me/top/tracks?time_range=short_term&limit=40'
    bearer = 'BQByACtQiEM_KLu6jopkLnWTNXyBWbwYni1WOC-4VRRmN7EbBqDfGokOa71nlryRxC3v6xycNwE39zrxfIl0f6Vt2kcOY2YunTt-V_Pm75HGvWD2yyDnReTCBIJ1Uqd3RnQNXG4mwQWO6XoGisAuzrUtIVhv9Ced-don1WY'
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
                print(audio_url)
                print(item['name'])
                doc = requests.get(item['preview_url'])
                with open("current_track/" + item['artists'][0]['name'] + "_" + re.sub("[:*<>|?]", " ", item['name'])+'.mp3', 'wb') as f:
                    f.write(doc.content)

            else:
                print('Link for audio '+track_name+' wasn\'t found')
        return 1
    else:
        print('Request failed, code ' + str(response.status_code))
        return 0
