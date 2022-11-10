import cv2
import numpy as np

from .utils import *

# 画像の上限画素数
MAX_WIDTH = 500
MAX_HEIGHT = 500


class ImagePreparer():
    def __init__(self, src_image: cv2.Mat, step: int) -> None:
        self.src_image = src_image
        self.step = step


    def _resize(self) -> None:
        '''
        input画像をリサイズ
        縦横共に500px以内のサイズに収める
        '''
        h, w, _ = self.src_image.shape
        if w > MAX_WIDTH:
            resized_h = round(h * (MAX_WIDTH / w))
            self.src_image = cv2.resize(self.src_image, dsize=(MAX_WIDTH, resized_h))
        if h > MAX_HEIGHT:
            resized_w = round(w * (MAX_HEIGHT / h))
            self.src_image = cv2.resize(self.src_image, dsize=(resized_w, MAX_HEIGHT))


    def _gray_scale(self) -> None:
        '''
        input画像を白黒画像に変換
        '''
        self.src_image = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2GRAY)


    def _mosaic(self, ratio: float=0.5) -> None:
        '''
        input画像にモザイク処理をかけ、タイル画像との重ね合わせ相性をよくする
        一度縮小して解像度を落とした画像を元のサイズに戻すことで実現
        '''
        small_image = cv2.resize(self.src_image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)   # type: ignore
        self.src_image = cv2.resize(small_image, self.src_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


    def _binarization(self) -> None:
        '''
        画像を指定範囲ごとに2値化してtmpディレクトリに出力
        '''
        for i in range(self.step):
            # 各ステップでの二値化範囲を指定
            lower_thresh = int(230 / self.step * i)
            upper_thresh = int(230 / self.step * (i+1))

            lower = np.array([lower_thresh])
            upper = np.array([upper_thresh])  

            binary_img = cv2.inRange(self.src_image, lower, upper)
            binary_img = cv2.bitwise_not(binary_img)

            cv2.imwrite(f'./img/tmp/input/input_{i}.png', binary_img)


    def run(self) -> tuple[int, int]:
        self._resize()
        self._gray_scale()
        self._mosaic()
        self._binarization()

        x, y = self.src_image.shape
        return x, y
