import cv2
from IPython.display import display, Image


def showImage(image, format=".jpg"):
    data = cv2.imencode(format, image)[1].tobytes()
    display(Image(data))


def thumnail(image, scale=10):
    height, width, _ = image.shape
    return cv2.resize(image, dsize=(width//scale, height//scale))
