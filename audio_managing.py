import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
cid = '032bb2c730e645968318b1811d084943'
secret = '5892fdf2a41948e1ae73078e4cecb6f2'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
artist_name = []
track_name = []
popularity = []
track_id = []
for i in range(0,100,50):
    track_results = sp.search(q='year:2018', type='track', limit=50,offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        popularity.append(t['popularity'])
print(track_results)
