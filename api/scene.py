import cv2
import openface
import pipeline as pl
import StringIO
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class Face:
    def __init__(self, name, bb=None, landmarks=None, phash=None, v128=None):
        self.bb = bb
        self.landmarks = landmarks
        self.phash = phash
        self.v128 = v128
        self.name = name


class Scene:
    def __init__(self):
        self.faces = []
        # self.images[phash] = Face(rep, identity)

    def new(self, bbs, all_landmarks, phashes, v128s):
        for i, bb in enumerate(bbs):
            landmarks = all_landmarks[i] if all_landmarks else None
            phash = phashes[i] if phashes else None
            v128 = v128s[i].tolist() if v128s else None
            bb = [bb.left(), bb.top(), bb.right(), bb.bottom()]
            self.faces.append(Face(str(i), bb, landmarks, phash, v128))
        return self


def all_faces(picture, do_landmarks=False, do_phash=False, do_v128=False, do_annotate=False):
    rgb_frame, buf = pl.stream2rgb_frame(picture)
    bbs = pl.all_face_bounding_boxes(rgb_frame)
    all_landmarks = None
    phashes = None
    png_stream = None
    v128s = None
    if do_landmarks or do_phash or do_v128 or do_annotate:
        all_landmarks = map(lambda bb: pl.find_landmarks(rgb_frame, bb), bbs)
        if do_phash or do_v128 or do_annotate:
            aligned_faces = map(lambda face: pl.align_face(rgb_frame, face[0], face[1]), zip(bbs, all_landmarks))
            if do_phash:
                phashes = list(map(pl.get_phash, aligned_faces))
            if do_v128 or do_annotate:
                v128s = map(pl.v128, aligned_faces)
                if do_annotate:
                    annotate(buf, bbs, all_landmarks, v128s)
                    png_stream = frame2stream(buf)
    return Scene().new(bbs, all_landmarks, phashes, v128s), png_stream


def annotate(frame, bbs, all_landmarks, v128s):
    for i, bb in enumerate(bbs):
        # name = str(v128s[i])
        name = "[ Unknown_" + str(i) + " ]"
        landmarks = all_landmarks[i]
        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
        cv2.rectangle(frame, bl, tr, color=(153, 255, 204),
                      thickness=3)
        for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
            cv2.circle(frame, center=landmarks[p], radius=3,
                       color=(102, 204, 255), thickness=-1)
        cv2.putText(frame, name, (bb.left(), bb.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                    color=(152, 255, 204), thickness=2)
    return frame


def frame2stream(frame):
    plt.figure()
    plt.imshow(frame)
    plt.xticks([])
    plt.yticks([])
    img = StringIO.StringIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img
