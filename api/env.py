import os
import sys

imgDim = 96

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", "openface"))

modelDir = os.path.join(fileDir, '..', "openface", 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

openfaceModelDir = os.path.join(modelDir, 'openface')
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
# cuda = True
cuda = False

