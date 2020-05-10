# face_detect
train/test/eval face detect model from winderface dataset

# model
I reference yolo(specialy yolov3) and ssd model,and simply these model.  
![](https://github.com/sunnythree/face_detect/blob/master/doc/model.png)  
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
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic1.png)  
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic2.png)  
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic3.png)  
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic4.png)  
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic5.png)  
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic6.png)  
![](https://github.com/sunnythree/face_detect/blob/master/doc/pic7.png)  
  

### I am still training and optimize the model