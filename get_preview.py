import json
import requests
import json
import re


def top_tracks_information(time_range='short_term'):
    top_url = 'https://api.spotify.com/v1/me/top/tracks?time_range=' + \
        time_range + '&limit=20'
    bearer = 'BQBGMT4swLYY7Xq-yiaOsoo6XXj2loUEXAkDHHoxMvKOrmYP_RZOxaB89ANxPSN6I0okrPEVbWpWxyEVasVxHXp4mqNB55ejbPLHiTL7e31N24hQxKZGNc1xuY9sBNuHo6Br748cZs69QgcV1O_KO9f6TAJdsFUUIzsAHNQ'
    headers = {'Authorization': 'Bearer ' + bearer}
    response = requests.get(top_url, headers=headers)
    if response.status_code == 200:
        print('Received track metadata successfully, code ' +
              str(response.status_code))
        parsed = json.loads(response.text)
        items = parsed['items']
        if items == [] and time_range == 'short_term':
            top_tracks_information('medium_term')
        elif items == [] and time_range == 'medium_term':
            top_tracks_information('long_term')
        elif items == [] and time_range == 'long_term':
            return 0
        for item in items:
            track_name = item['name']
            artist_name = item['artists'][0]['name']
            if "preview_url" in item and item['preview_url'] != None:
                audio_url = item['preview_url']
                # print(audio_url)
                # print(item['name'])
                doc = requests.get(item['preview_url'])
                with open("current_track/" + item['artists'][0]['name'] + "_" + re.sub("[:*<>|?]", " ", item['name'])+'.mp3', 'wb') as f:
                    f.write(doc.content)

            else:
                print('Link for audio '+track_name+' wasn\'t found')
        return 1
    else:
        print('Request failed, code ' + str(response.status_code))
        return 0
