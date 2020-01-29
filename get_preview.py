import urllib.request
import json
import urllib.parse
import requests
import json


def preview_download():
    track_url = 'https://api.spotify.com/v1/tracks/3rCtueI7qBN2kZBZnXuk5K'
    headers = {'Authorization': 'Bearer BQAIVVr2KXnCl5p3fsZFOVbgT5L3ELMQtj2nKnFe4ISBk-gY9KAklHZKYjx_rhijTwWmH896vP5zx7pmJL8VHh35ii2YL9S3p7JPQlNOY_IDF_07OINauXDoWaOQuDsBJzwe5_R63V2fd_KbWLgelwfjEfJUeL0'}
    r = requests.get(track_url, headers=headers)
    parsed = json.loads(r.text)
    track_name = parsed['name']
    artist_name = parsed['artists'][0]['name']
    audio_url = parsed['preview_url']
    urllib.request.urlretrieve(
        audio_url,  "current_track/" + artist_name+'_'+track_name+'.mp3')


preview_download()
