import json
from flask import Flask, request, flash, redirect, send_file

import pipeline as pl
import scene

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def get_picture_stream(req):
    # check if the post request has the file part
    if 'picture' not in req.files:
        # flash('No file part')
        return None
    picture = req.files['picture']
    # if user does not select file, browser also
    # submit a empty part without filename
    if picture.filename == '':
        # flash('No selected file')
        return None
    if picture:
        return picture.stream
    return None


@app.route('/getAllFaceBoundingBoxes', methods=['GET', 'POST'])
def get_all_face_bounding_boxes():
    if request.method == 'POST':
        picture = get_picture_stream(request)
        if picture is None:
            return redirect(request.url)
        rgb_frame, _ = pl.stream2rgb_frame(picture)
        bbs = pl.all_face_bounding_boxes(rgb_frame)
        msg = {
            "bb": pl.dlib_rectangles2array(bbs)
        }
        rsp = json.dumps(msg)
        print(rsp)
        return rsp
    return '''
    <!doctype html>
    <title>getAllFaceBoundingBoxes</title>
    <h1>getAllFaceBoundingBoxes</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=picture>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/getFacesMeta', methods=['GET', 'POST'])
def get_faces_meta():
    if request.method == 'POST':
        picture = get_picture_stream(request)
        if picture is None:
            return redirect(request.url)
        do_landmarks = True if request.form.get('do_landmarks') else False
        do_phash = True if request.form.get('do_phash') else False
        do_v128 = True if request.form.get('do_v128') else False
        meta, _ = scene.all_faces(picture, do_landmarks, do_phash, do_v128)
        rsp = to_json(meta)
        print("Rsp: size=" + str(len(rsp)) + ", data={" + rsp + "}")
        return rsp
    return '''
    <!doctype html>
    <title>getFacesMeta</title>
    <h1>getFacesMeta</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=picture>
      <p><input type=checkbox name=do_landmarks> do_landmarks
      <p><input type=checkbox name=do_phash> do_phash
      <p><input type=checkbox name=do_v128> do_v128
      <p><input type=submit value=Upload>
    </form>
    '''

@app.route('/getAllFacesMeta', methods=['GET', 'POST'])
def get_all_faces_meta():
    if request.method == 'POST':
        picture = get_picture_stream(request)
        if picture is None:
            return redirect(request.url)
        meta, _ = scene.all_faces(picture, do_phash=True, do_v128=True)
        rsp = to_json(meta)
        print("Rsp: size=" + str(len(rsp)) + ", data={" + rsp + "}")
        return rsp
    return '''
    <!doctype html>
    <title>getAllFacesMeta</title>
    <h1>getAllFacesMeta</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=picture>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/getAllFaces', methods=['GET', 'POST'])
def get_all_faces():
    if request.method == 'POST':
        picture = get_picture_stream(request)
        if picture is None:
            return redirect(request.url)
        _, png_stream = scene.all_faces(picture, do_annotate=True)
        return send_file(png_stream, mimetype='image/png')
    return '''
    <!doctype html>
    <title>getAllFaces</title>
    <h1>getAllFaces</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=picture>
         <input type=submit value=Upload>
    </form>
    '''


def to_json(sc):
    return json.dumps(sc, default=lambda o: o.__dict__, sort_keys=True)

"""
 if __name__ == "__main__":
 app.run(threaded=True, host="0.0.0.0")
"""
