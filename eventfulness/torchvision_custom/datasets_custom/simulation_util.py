import numpy as np
import copy

def writePixelinFrame(frame, hi, wi, value):
    copyF = copy.deepcopy(frame)
    copyF[hi, wi, :] = value
    return copyF

def generateRandomIllumFrame(h, w, c):
    illum = np.random.randint(0, 256, size=(h,w,1), dtype=np.uint8)
    frame = illum
    for i in range(1,c):
         frame = np.concatenate((frame, illum), axis=2)
    return frame

def generateRandomIndex(h, w):
    hi = np.random.randint(0, h, 1, dtype=np.uint8)
    wi = np.random.randint(0, w, 1, dtype=np.uint8)
    return (hi, wi)

def makeRandomBinaryVector(n):
    return  np.random.randint(0, 2, (n), dtype=np.uint8)

def make_random_flicker_video(frame_n, h, w, c):
    labels = makeRandomBinaryVector(frame_n)
    normalFrame = generateRandomIllumFrame(h, w, c)
    (hi, wi) = generateRandomIndex(h, w)
    randValue1 = np.random.randint(0, 255, 1, dtype=np.uint8)
    randValue2 = np.random.randint(0, 255, 1, dtype=np.uint8)
    randColorBright = randValue1[0]
    randColorDark = randValue2[0]
    video = np.zeros((frame_n, h, w, c), dtype=np.uint8)

    if randValue2[0] > randValue1[0]:
        randColorBright = randValue2[0]
        randColorDark = randValue1[0]

    for i in range(frame_n):
        label = labels[i]
        # print("label i {}: {}".format(i, label))
        if( label > 0 ):
            video[i, :, :, :] = writePixelinFrame(normalFrame, hi, wi, randColorBright)
        else:
            video[i, :, :, :] = writePixelinFrame(normalFrame, hi, wi, randColorDark)

    return video, labels