__author__ = 'tz3'

import cv2

import config

faceCascade = cv2.CascadeClassifier(config.HAARCASCADE)
eye_cascade = cv2.CascadeClassifier(config.HAARCASCASE_EYE)


def detect(image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=1,
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                          2)
    return image


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    w = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    # video doesn't play
    out = cv2.VideoWriter(
        config.result_path("%s.mp4" % config.current_filename(__file__)), -1,
        25,
        (w, h))
    while 1:
        _, frame = cam.read()
        frame = detect(frame)
        if cam.isOpened():
            out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == 0x1b:  # ESC
            print 'ESC pressed. Exiting ...'
            break
    cam.release()
    out.release()
    cv2.destroyAllWindows()
