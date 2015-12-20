# some filters
#
# "0 - Read image",
# "1 - Apply linear filter",
# "2 - Apply blur(...)",
# "3 - Apply medianBlur(...)",
# "4 - Apply GaussianBlur(...)",
# "5 - Apply erode(...)",
# "6 - Apply dilate(...)",
# "7 - Apply Sobel(...)",
# "8 - Apply Laplacian(...)",
# "9 - Apply Canny(...)",
# "10 - Apply calcHist(...)",
# "11 - Apply equalizeHist(...)"
import cv2

from config import config


class Image(object):
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def linear_filter(self):
        pass

    def gray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur(self):
        self.image = cv2.blur(self.image, (5, 5))

    def median_blur(self):
        self.image = cv2.medianBlur(self.image, 5)

    def gaussian_blur(self):
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)

    def erode(self):
        self.image = cv2.erode(self.image, (200, 200))

    def dilate(self):
        self.image = cv2.dilate(self.image, (200, 200))

    def sobel(self):
        self.image = cv2.Sobel(self.image, cv2.CV_64F, 1, 1, ksize=5)

    def laplacian(self):
        self.image = cv2.Laplacian(self.image, cv2.CV_64F)

    def canny(self):
        self.image = cv2.Canny(self.image, 10, 100, 3)

    def calc_hist(self):
        self.image = cv2.calcHist([self.image], [0], None, [256], [0, 256])

    def equalize_hist(self):
        self.gray()
        self.image = cv2.equalizeHist(self.image)

    def get_image(self):
        return self.image


# i = Image(config.ORIGIN_IMAGE)
# image = cv2.imread(config.ORIGIN_IMAGE)
# cv2.imshow('something', i.get_image())
# cv2.waitKey(0)
# i.equalize_hist()
# cv2.imshow('something', i.get_image())
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == "__main__":
    print config.ASSET_DIR