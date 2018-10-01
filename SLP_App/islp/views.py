from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug import secure_filename
from islp import app
import os
import librosa as lib
import pandas as pd
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


def plotwarp(D, wp, hop_size, fs, putpath):
    wp_s = np.asarray(wp) * hop_size / fs

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    lib.display.specshow(D, x_axis='time', y_axis='time',
                         cmap='viridis', hop_length=hop_size)
    imax = ax.imshow(D, cmap=plt.get_cmap('viridis'),
                     origin='lower', interpolation='nearest', aspect='auto')

    ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
    plt.title('Warping Path on Cost Matrix')
    plt.colorbar()
    savefig(putpath)
    plt.gcf().clear()
    return putpath


def plotmatch(x_1, fs1, x_2, fs2, wp, hop_size, putpath):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    # Plot x_1
    lib.display.waveplot(x_1, sr=fs1, ax=ax1)
    ax1.set(title='Audio Template')

    # Plot x_2
    lib.display.waveplot(x_2, sr=fs2, ax=ax2)
    ax2.set(title='Uploaded Audio File')

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


def get_warp_features(D, wp):
    wp_df = pd.DataFrame(wp, columns=['fixed_index', 'moving_index'])

    warp_features = pd.DataFrame(wp_df.fixed_index.value_counts())
    warp_features.rename(
        index=str, columns={'fixed_index': 'fixed_frequency'}, inplace=True)
    warp_features.index = warp_features.index.astype(int)
    warp_features.sort_index(inplace=True)

    fixed_vc = pd.DataFrame(wp_df.fixed_index.value_counts())
    fixed_vc.rename(
        index=str, columns={'fixed_index': 'fixed_frequency'}, inplace=True)

    moving_vc = pd.DataFrame(wp_df.moving_index.value_counts())
    moving_vc.rename(
        index=str, columns={'moving_index': 'moving_frequency'}, inplace=True)

    moving_vc.index = moving_vc.index.astype(int)
    freq_df = wp_df.join(moving_vc, on='moving_index')
    freq_df.drop_duplicates(subset='fixed_index', inplace=True)

    freq_df2 = warp_features.join(freq_df.set_index('fixed_index'))
    freq_df2.drop(columns='moving_index', inplace=True)

    freq_df2[
        'local_warp'] = freq_df2.fixed_frequency / freq_df2.moving_frequency
    freq_df2.drop(columns=['fixed_frequency',
                           'moving_frequency'], inplace=True)

    point2point_df = pd.DataFrame(wp, columns=['fixed_index', 'moving_index'])

    temp = []
    for i in range(0, wp.shape[0]):
        temp.append(D[wp[i, 0], wp[i, 1]])
    point2point_df['cost_func'] = temp

    costfunc_feature = pd.DataFrame(
        point2point_df.groupby('fixed_index')['cost_func'].mean())

    warp_final_features = freq_df2.join(costfunc_feature)

    return warp_final_features


def process_case(wav_path, demo=False, hop_size=512,
                 trim_path_template='islp/models/kesshi_grandfather_trim.wav',
                 model_path='islp/models/svc_binary_kesh_template.pkl'):
    # load and display the uploaded waveform
    if not demo:
        y, sr = loadwav(wav_path)
    wavpng = wav_path.replace('.wav', '_wav.png')
    if not demo:
        plotwave(y, sr, wavpng)
    # trim the uploaded waveform (remove silence at beginning or end)
    trim_wav_path = wav_path.replace('.wav', '_trim.wav')
    print(trim_wav_path)
    if not demo:
        save_trimwav(wav_path, trim_wav_path)
        y, sr = loadwav(trim_wav_path)
    trimpng = trim_wav_path.replace('.wav', '_wav.png')
    if not demo:
        plotwave(y, sr, trimpng)

        # Load template file
        y_template, sr_template = loadwav(trim_path_template)

        # Warp the uploaded file to the template
        print('warping')
        D, wp = dowarp_mfcc(y_template, y, sr_template, sr)
        print(D.shape)
        print(wp.shape)

    warppng = trim_wav_path.replace('.wav', '_warp.png')
    matchpng = trim_wav_path.replace('.wav', '_match.png')
    if not demo:
        plotwarp(D, wp, hop_size, sr_template, warppng)
        plotmatch(y_template, sr_template, y, sr,
                  wp, hop_size, matchpng)

        # Extract features
        features = get_warp_features(D, wp)

    csv_path = trim_wav_path.replace('.wav', '_features.csv')

    if not demo:
        features.to_csv(csv_path)
    else:
        features = pd.read_csv(csv_path, index_col=0)

    print('FEATURES')
    print(features.shape)

    # Load and run the model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    feature_vec = np.transpose(np.array(features).flatten('F').reshape(-1, 1))
    print('FEATURE VEC')
    print(feature_vec.shape)
    print(feature_vec[0:5, :])
    prediction = loaded_model.predict(feature_vec)[0]
    P_class1, P_class2 = loaded_model.predict_proba(feature_vec)[0]
    print('PREDICT')
    print(loaded_model.predict(feature_vec))
    print('PROBABILITIES')
    print(P_class1, P_class2)
    print('CLASSES')
    print(loaded_model.classes_)
    model_output = {'Probabilities': list([P_class1, P_class2]),
                    'Classes': list(loaded_model.classes_),
                    'Prediction': prediction}

    return wavpng, trimpng, warppng, matchpng, csv_path, model_output


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
        ext = os.path.splitext(filename)[1]
        if (ext == ".wav"):
            print("File accepted")
        else:
            return render_template("error.html", message="only .wav files are supported at this time.")
        putpath_wav = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(putpath_wav)
        print(os.path.abspath(putpath_wav))

        wavpng, trimpng, warppng, matchpng, csv, model_dict = process_case(
            putpath_wav)

        wavpng_path = os.path.join(
            '/static/images/', os.path.basename(wavpng))
        trimpng_path = os.path.join(
            '/static/images/', os.path.basename(trimpng))
        warppng_path = os.path.join(
            '/static/images/', os.path.basename(warppng))
        matchpng_path = os.path.join(
            '/static/images/', os.path.basename(matchpng))
        csv_path = os.path.join(
            '/static/images/', os.path.basename(csv))
        session['wav_path'] = putpath_wav
        session['wav_png_path'] = wavpng_path
        session['trim_png_path'] = trimpng_path
        session['warp_png_path'] = warppng_path
        session['match_png_path'] = matchpng_path
        session['csv_path'] = csv_path
        session['model_dict'] = model_dict
        return render_template("loaded_raw.html",
                               wavfile=putpath_wav, wavpng=wavpng_path)


@app.route('/demo', methods=['GET', 'POST'])
def run_demo():
    print('DEMOing')
    if request.method == 'POST':
        putpath_wav = 'islp/static/images/Demo_Negative.wav'

        wavpng, trimpng, warppng, matchpng, csv, model_dict = process_case(
            putpath_wav, demo=True)

        wavpng_path = os.path.join(
            '/static/images/', os.path.basename(wavpng))
        trimpng_path = os.path.join(
            '/static/images/', os.path.basename(trimpng))
        warppng_path = os.path.join(
            '/static/images/', os.path.basename(warppng))
        matchpng_path = os.path.join(
            '/static/images/', os.path.basename(matchpng))
        csv_path = os.path.join(
            '/static/images/', os.path.basename(csv))
        session['wav_path'] = putpath_wav
        session['wav_png_path'] = wavpng_path
        session['trim_png_path'] = trimpng_path
        session['warp_png_path'] = warppng_path
        session['match_png_path'] = matchpng_path
        session['csv_path'] = csv_path
        session['model_dict'] = model_dict

        return render_template("demo.html",
                               wavfile=putpath_wav, wavepng=wavpng_path)


@app.route('/demo_positive', methods=['GET', 'POST'])
def run_demo_pos():
    if request.method == 'POST':
        print('POSTING')
        putpath_wav = 'islp/static/images/Demo_Positive.wav'
        print(os.path.abspath(putpath_wav))

        wavpng, trimpng, warppng, matchpng, csv, model_dict = process_case(
            putpath_wav, demo=True)

        wavpng_path = os.path.join(
            '/static/images/', os.path.basename(wavpng))
        trimpng_path = os.path.join(
            '/static/images/', os.path.basename(trimpng))
        warppng_path = os.path.join(
            '/static/images/', os.path.basename(warppng))
        matchpng_path = os.path.join(
            '/static/images/', os.path.basename(matchpng))
        csv_path = os.path.join(
            '/static/images/', os.path.basename(csv))
        session['wav_path'] = putpath_wav
        session['wav_png_path'] = wavpng_path
        session['trim_png_path'] = trimpng_path
        session['warp_png_path'] = warppng_path
        session['match_png_path'] = matchpng_path
        session['csv_path'] = csv_path
        session['model_dict'] = model_dict
        return render_template("demo.html",
                               wavfile=putpath_wav, wavepng=wavpng_path)


@app.route('/qc')
def qc():
    # wav_path = session.get('wav_path', None)
    wav_png = session.get('wav_png_path', None)
    trim_png = session.get('trim_png_path', None)

    user = {'PIDN': '1234'}
    snapshots = [{'name': 'Original Waveform', 'path': wav_png},
                 {'name': 'Trimmed Waveform', 'path': trim_png}]

    return render_template("qc.html", title='QC', user=user, snapshots=snapshots)


@app.route('/warp_qc')
def warp_qc():

    warp_png_path = session.get('warp_png_path', None)
    match_png_path = session.get('match_png_path', None)

    user = {'PIDN': '1234'}
    snapshots = [{'name': 'Warp to Template Waveform', 'path': warp_png_path},
                 {'name': 'Matched Waveforms', 'path': match_png_path}]

    return render_template("warp_qc.html", title='Warp_QC', user=user, snapshots=snapshots)


@app.route('/model')
def model():

    model_dict = session.get('model_dict', None)

    user = {'PIDN': '1234'}
    diagnosis = {'fullname': model_dict['Prediction']}
    posts = [
        {
            'group': model_dict['Classes'][0],
            'probability': "{:.2%}".format(model_dict['Probabilities'][0])
        },
        {
            'group': model_dict['Classes'][1],
            'probability': "{:.2%}".format(model_dict['Probabilities'][1])
        },
    ]

    return render_template("model.html", title='Prediction', user=user, posts=posts, diagnosis=diagnosis)
