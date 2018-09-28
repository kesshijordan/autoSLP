from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from islp import app
import os

import librosa as lib
import librosa.display as libd
# import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt


def plotwave(y, sr, putpath):
    libd.waveplot(y, sr=sr, x_axis='time')
    savefig(putpath)
    plt.gcf().clear()
    return putpath


def loadwav(wavpath):
    y, sr = lib.load(wavpath)
    S_full, phase = lib.magphase(lib.stft(y))
    return y, sr, S_full, phase


@app.route('/')
@app.route('/index')
def index():
    user = {'PIDN': '1234'}
    snapshots = [{'name': 'waveform', 'path': '/static/images/test1.png'},
                 {'name': 'spectrogram', 'path': '/static/images/test2.png'}]
    return render_template("index.html", title='Home', user=user, snapshots=snapshots)


@app.route('/upload')
def upload():
    return render_template("upload.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        '''ext = os.path.splitext(filename)[1]
        if (ext == ".wav"):
            print("File accepted")
        else:
            return render_template("error.html", message="Only .wav files are supported at this time."), 400'''
        putpath_wav = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(putpath_wav)
        print(os.path.abspath(putpath_wav))
        # return redirect('/qc')

        # Next we we want to load and display the waveform
        y, sr, S_full, phase = loadwav(putpath_wav)
        wavepng = plotwave(y, sr, putpath_wav.replace('.wav', '_wav.png'))
        wavepng_path = os.path.join(
            '/static/images/', os.path.basename(wavepng))
        return render_template("loaded_raw.html",
                               wavfile=putpath_wav, wavepng=wavepng_path)


@app.route('/qc')
def qc():
    user = {'PIDN': '1234'}
    snapshots = [{'name': 'waveform', 'path': '/static/images/test1.png'},
                 {'name': 'spectrogram', 'path': '/static/images/test2.png'}]

    return render_template("qc.html", title='QC', user=user, snapshots=snapshots)


@app.route('/model')
def model():
    user = {'PIDN': '1234'}
    diagnosis = {'fullname': 'Nonfluent Variant Primary Progressive Aphasia'}
    posts = [
        {
            'group': 'Normal Aging',
            'probability': '10%'
        },
        {
            'group': 'Nonfluent Variant PPA',
            'probability': '60%'
        },
        {
            'group': 'Semantic Variant PPA',
            'probability': '16%'
        },
        {
            'group': 'Logopenic Variant PPA',
            'probability': '14%'
        }
    ]

    return render_template("model.html", title='Prediction', user=user, posts=posts, diagnosis=diagnosis)
