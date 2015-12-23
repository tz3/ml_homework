__author__ = 'tz3'
# make filter on picture
# blur
# boxFilter
# gaussianBlur
# medianBlur

import cv2

from config import config


def filters():
    image = cv2.imread(config.ORIGIN_IMAGE)
    blur = [cv2.GaussianBlur(image, (5, 5), 0)]
    blur.append(cv2.medianBlur(image, 5))
    blur.append(cv2.boxFilter(image, 5, (5, 5)))
    for i, x in enumerate(blur):
        cv2.imwrite(
            config.RESULT_PATH + "%s_mode_%s.jpg" % (
                config.current_filename(__file__), i),
            x)


if __name__ == "__main__":
    # video doesn't play
    filters()
