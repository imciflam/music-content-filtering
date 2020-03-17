# Content-based filtering for music recommendation system

Using CNN, RNN and CRNN to classify music by genre & to create recommendations.

### for dev purposes: 
install dependencies 

```pip(3) install -r requirements.txt```

install ffmpeg (don't forget to add link to it to your PATH env variable)

then run

```python index.py```  

### known issues
if you are getting ```Tensor Tensor("dense_2/Softmax:0", shape=(?, 10), dtype=float32) is not an element of this graph.``` error, try adding  ```model._make_predict_function()``` after loading the model
