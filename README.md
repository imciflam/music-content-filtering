# Content-based filtering for music recommendation system

### Description: 
This server is using CNN, RNN and CRNN models to classify music by genre & to create personal recommendations utilising either cosine distance or Jensen-Shannon distance. Part of Anitra music recommendation system.

### Installation (for dev purposes):
1. Install dependencies 

```pip(3) install -r requirements.txt```

2. Install [ffmpeg](https://www.ffmpeg.org/download.html) (don't forget to add link to it to your PATH env variable)

3. Run

```python index.py```  

### known issues
if you are getting ```Tensor Tensor("dense_2/Softmax:0", shape=(?, 10), dtype=float32) is not an element of this graph.``` error, try adding  ```model._make_predict_function()``` after loading the model
