import urllib.request
import json
import urllib.parse
import requests
import json

url = 'https://api.spotify.com/v1/tracks/3rCtueI7qBN2kZBZnXuk5K'
headers = {'Authorization': 'Bearer BQAF-BhPdxQdQzJ0Ix-XTlIQJ9tn8qVl3UfBJxR_tD2ew5DRzZ8AIK7My3vNFZ0Yo2MKs7orYKuEUUd1dDemBUpEYNB8QPO1EgFWliTeio9XipMra6TrHYSwhO7Ndpa4qNCO3CI0aLx3mlUSEc_bBBAmsePqSSw'}
r = requests.get(url, headers=headers)
parsed = json.loads(r.text)
print(parsed['preview_url'])


url = 'https://p.scdn.co/mp3-preview/a5edf2b1a999be17263c35ea296cb41d2b72ad25?cid=774b29d4f13844c495f206cafdad9c86'
urllib.request.urlretrieve(url, 'kek.mp3')
