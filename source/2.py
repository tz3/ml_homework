import cv2

import config

img1 = cv2.imread(config.ORIGIN_IMAGE)
imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 200, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(thresh, contours, 30, (0, 255, 0), 3)
cv2.imwrite(config.RESULT_PATH + '2.jpg', thresh)
# cv2.waitKey(0)
cv2.destroyAllWindows()
