# face_detect
train/test/eval face detect model from winderface dataset

# model
I reference yolo(specialy yolov3) and ssd model,and simply these model.

# train
```python3.6 train.py -l 0.001 -e 10 -b 5```  
detailes: -l for learning rate, -e for epoes, -b for batch.  
if you want train from last model, add -p True, like this:
```python3.6 train.py -l 0.001 -e 10 -b 5 -p True``` 

# test
```python3.6 test.py```  

# eval
```python3.6 eval.py -c 0.6 -t 0.5```  
details: -c for the confidence of box that contains face.  
-t for thresh of IOU


# shows
[result]()  

### I am still training and optimize the model