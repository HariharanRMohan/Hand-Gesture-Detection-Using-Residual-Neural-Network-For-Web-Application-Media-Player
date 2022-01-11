from flask import Flask, render_template, Response
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from werkzeug.wrappers import response
from keyboard import Keyboard


app = Flask(__name__)
vid = cv2.VideoCapture(0)
model=load_model(r'D:/HandGestureRecognitionFYP1/model/resnetmodel.hdf5')
labels = pd.read_csv(r'D:/fyp datasets/2OBN JESTER/jester-v1-labels.csv', header= None)


def generate_frames():
    
    buffer = []
    cls = []
    predicted_value = 0
    final_label = ""
    i = 1

    # Check if the webcam is opened correctly
    if not vid.isOpened():
        raise IOError("Cannot open webcam")

    while (vid.isOpened()):
        ret,frame = vid.read()
        if ret:
            image = cv2.resize(frame,(96,64))
            image = image/255.0
            buffer.append(image)
            if(i%16==0):
                buffer = np.expand_dims(buffer,0)
                predicted_value =np.argmax(model.predict(buffer))
                cls = labels.iloc[predicted_value]
                print(cls)
                print(cls.iloc[0])
                if(predicted_value == 0):
                    final_label = "Swiping left"
                    Keyboard.key(Keyboard.VK_MEDIA_PREV_TRACK)
                elif (predicted_value == 1):
                    final_label = "Swiping right"
                    Keyboard.key(Keyboard.VK_MEDIA_NEXT_TRACK)
                elif (predicted_value == 8):
                    final_label = "sliding two fingres down"
                    Keyboard.key(Keyboard.VK_VOLUME_DOWN)
                elif (predicted_value== 9):
                    final_label = "sliding two fingres up"
                    Keyboard.key(Keyboard.VK_VOLUME_UP)
                elif (predicted_value == 23):
                    final_label = "stop sign"
                    Keyboard.key(Keyboard.VK_VOLUME_MUTE)
                elif (predicted_value == 24):
                    final_label = "drumming fingers"
                    Keyboard.key(Keyboard.VK_MEDIA_PLAY_PAUSE)
                elif (predicted_value == 25):
                    final_label = "no gesture"
                else:
                    final_label = "doing other things"
                cv2.imshow('frame',frame)
                buffer = []
            i = i+1
            text = "activity: {}".format(final_label)
            cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,1.15, (0, 255, 0), 5) 
            cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


@app.route('/')
def homepage():
    return render_template('indextest.html')

@app.route('/guidepage')
def guidepage():
    return render_template('guidetest.html')

@app.route('/aboutpage')
def aboutpage():
    return render_template('abouttest.html')

@app.route('/guidetohome')
def guidetohome():
    return render_template('indextest.html')

@app.route('/video') ##define video func which return something from here
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)


   