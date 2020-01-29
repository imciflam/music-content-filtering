import urllib.request
import json
import urllib.parse
import requests
import json

track_url = 'https://api.spotify.com/v1/tracks/3rCtueI7qBN2kZBZnXuk5K'
headers = {'Authorization': 'Bearer BQAF-BhPdxQdQzJ0Ix-XTlIQJ9tn8qVl3UfBJxR_tD2ew5DRzZ8AIK7My3vNFZ0Yo2MKs7orYKuEUUd1dDemBUpEYNB8QPO1EgFWliTeio9XipMra6TrHYSwhO7Ndpa4qNCO3CI0aLx3mlUSEc_bBBAmsePqSSw'}
r = requests.get(track_url, headers=headers)
parsed = json.loads(r.text)
track_name = parsed['name']
artist_name = parsed['artists'][0]['name']
audio_url = parsed['preview_url']
urllib.request.urlretrieve(audio_url, artist_name+'_'+track_name+'.mp3')
