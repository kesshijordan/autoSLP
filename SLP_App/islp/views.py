from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug import secure_filename
from islp import app
import os
import librosa as lib
import numpy as np
import librosa.display as libd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import pickle


def loadwav(wavpath):
    y, sr = lib.load(wavpath)
    # S_full, phase = lib.magphase(lib.stft(y))
    return y, sr


def save_trimwav(wavpath, putpath, top_db=20):
    y, sr = lib.load(wavpath)
    yt, index = lib.effects.trim(y, top_db=top_db)
    print(lib.get_duration(y), lib.get_duration(yt))
    lib.output.write_wav(putpath, yt, sr)


def plotwave(y, sr, putpath):
    libd.waveplot(y, sr=sr, x_axis='time')
    savefig(putpath)
    plt.gcf().clear()
    return putpath


def plotwarp(D, wp, hop_size, fs, putpath):
    wp_s = np.asarray(wp) * hop_size / fs

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    lib.display.specshow(D, x_axis='time', y_axis='time',
                         cmap='gray_r', hop_length=hop_size)
    imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                     origin='lower', interpolation='nearest', aspect='auto')

    ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
    plt.title('Warping Path on Acc. Cost Matrix $D$')
    plt.colorbar()
    savefig(putpath)
    plt.gcf().clear()
    return putpath


def plotmatch(x_1, fs1, x_2, fs2, wp, hop_size, putpath):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    # Plot x_1
    lib.display.waveplot(x_1, sr=fs1, ax=ax1)
    ax1.set(title='Moving Version $X_1$')

    # Plot x_2
    lib.display.waveplot(x_2, sr=fs2, ax=ax2)
    ax2.set(title='Fixed Version $X_2$')

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx = np.int16(
        np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

    for tp1, tp2 in wp[points_idx] * hop_size / fs2:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = Line2D((coord1[0], coord2[0]),
                      (coord1[1], coord2[1]),
                      transform=fig.transFigure,
                      color='r')
        lines.append(line)

    fig.lines = lines
    plt.tight_layout()
    savefig(putpath)
    plt.gcf().clear()
    return putpath


def dowarp_mfcc(y1, y2, sr1, sr2):
    mfcc1 = lib.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = lib.feature.mfcc(y=y2, sr=sr2)
    D, wp = lib.sequence.dtw(X=mfcc1, Y=mfcc2, metric='cosine')
    # wp_s = np.asarray(melwp3) * hop_size / fs
    return D, wp


@app.route('/')
@app.route('/index')
def index():
    user = {'PIDN': '1234'}
    snapshots = [
        {'name': 'logo', 'path': '/static/images/iSLP_logo_small.png'}]
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
        y, sr = loadwav(putpath_wav)
        wavepng = plotwave(y, sr, putpath_wav.replace('.wav', '_wav.png'))
        wavepng_path = os.path.join(
            '/static/images/', os.path.basename(wavepng))
        session['wav_path'] = putpath_wav
        session['wav_png_path'] = wavepng_path
        return render_template("loaded_raw.html",
                               wavfile=putpath_wav, wavepng=wavepng_path)


@app.route('/qc')
def qc():
    wav_path = session.get('wav_path', None)
    wav_png = session.get('wav_png_path', None)
    print(wav_path)
    trim_wav_path = wav_path.replace('.wav', '_trim.wav')
    print(trim_wav_path)
    save_trimwav(wav_path, trim_wav_path)
    y, sr = loadwav(trim_wav_path)
    trimpng = plotwave(y, sr, trim_wav_path.replace('.wav', '_wav.png'))
    trimpng_path = os.path.join(
        '/static/images/', os.path.basename(trimpng))
    print('trimpng')
    print(trimpng)
    print(trimpng_path)
    user = {'PIDN': '1234'}
    snapshots = [{'name': 'Original Waveform', 'path': wav_png},
                 {'name': 'Trimmed Waveform', 'path': trimpng_path}]
    session['trim_path'] = trim_wav_path
    session['trim_png_path'] = trimpng_path
    return render_template("qc.html", title='QC', user=user, snapshots=snapshots)


@app.route('/warp_qc')
def warp_qc():
    trim_path_subject = session.get('trim_path', None)
    trim_path_template = 'islp/models/kesshi_grandfather_trim.wav'

    y_moving, sr_moving = loadwav(trim_path_subject)
    y_fixed, sr_fixed = loadwav(trim_path_template)

    print('warping')
    D, wp = dowarp_mfcc(y_fixed, y_moving, sr_fixed, sr_moving)

    hop_size = 512
    warp_png_putpath = trim_path_subject.replace('.wav', '_warp.png')
    match_png_putpath = trim_path_subject.replace('.wav', '_match.png')
    plotwarp(D, wp, hop_size, sr_fixed, warp_png_putpath)
    plotmatch(y_fixed, sr_fixed, y_moving, sr_moving,
              wp, hop_size, match_png_putpath)

    warppng_path = os.path.join(
        '/static/images/', os.path.basename(warp_png_putpath))
    matchpng_path = os.path.join(
        '/static/images/', os.path.basename(match_png_putpath))
    print('PATHS')
    print(warp_png_putpath)
    print(warppng_path)

    user = {'PIDN': '1234'}
    snapshots = [{'name': 'Warp to Template Waveform', 'path': warppng_path},
                 {'name': 'Matched Waveforms', 'path': matchpng_path}]

    return render_template("warp_qc.html", title='Warp_QC', user=user, snapshots=snapshots)


@app.route('/model')
def model():
    with open('islp/models/svc_binary.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print(loaded_model)
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
