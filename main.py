from flask import Flask, flash, redirect, render_template, request, session, abort,Response,url_for
from camera import VideoCamera
from camera2 import VideoCamera2
import os
app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('welcome.html')


@app.route('/stream',methods=['GET', 'POST'])
def index():
    # rendering webpage
    if request.method == 'POST':
        return redirect(url_for('index2'))
    return render_template('index.html')
def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


#2nd page
@app.route('/stream_face',methods=['GET', 'POST'])
def index2():
    # rendering webpage
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('index2.html')
def gen2(camera2):
    while True:
        # get camera frame
        frame2 = camera2.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
@app.route('/video_feed_face')
def video_feed_face():
    return Response(gen(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/login',methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'secret':
            error = 'Invalid Username/Password.'
        else:
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='localhost', port='5000', debug=True)
