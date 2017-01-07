import env
import numpy as np
from PIL import Image
import imagehash
import openface


img_dim = env.imgDim
dlib_face_predictor = env.dlibFacePredictor

align = openface.AlignDlib(dlib_face_predictor)
net = openface.TorchNeuralNet(env.networkModel, imgDim=env.imgDim, cuda=env.cuda)


def stream2rgb_frame(stream):
    image = Image.open(stream)
    buf = np.fliplr(np.asarray(image))
    print("Picture shape: " + str(buf.shape))
    w, h, c = buf.shape
    rgb_frame = np.zeros((w, h, c), dtype=np.uint8)
    rgb_frame[:, :, 0] = buf[:, :, 2]
    rgb_frame[:, :, 1] = buf[:, :, 1]
    rgb_frame[:, :, 2] = buf[:, :, 0]
    return rgb_frame, np.copy(buf)


def all_face_bounding_boxes(rgb_frame):
    return align.getAllFaceBoundingBoxes(rgb_frame)


def find_landmarks(rgb_frame, bb):
    return align.findLandmarks(rgb_frame, bb)


def align_face(rgb_frame, bb, landmarks):
    return align.align(img_dim, rgb_frame, bb,
                       landmarks=landmarks,
                       landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)


def get_phash(aligned_face):
    return str(imagehash.phash(Image.fromarray(aligned_face)))


def v128(aligned_face):
    return net.forward(aligned_face)


# helper function
def dlib_rectangles2array(rectangles):
    py_rects = []
    for r in rectangles:
        py_rects.append([r.left(), r.top(), r.right(), r.bottom()])
    return py_rects

