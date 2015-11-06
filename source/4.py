import cv2
import config
faceCascade = cv2.CascadeClassifier(config.HAARCASCADE)


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
    return image


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    while 1:
        _, frame = cam.read()
        frame = detect(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == 0x1b:  # ESC
            print 'ESC pressed. Exiting ...'
            cam.release()
            cv2.destroyAllWindows()
            break
