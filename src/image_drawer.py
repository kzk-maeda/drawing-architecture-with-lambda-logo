import os
from tkinter.messagebox import NO
import cv2
import glob
import numpy as np

from .utils import *


class ImageDrawer():
    def __init__(self, ratio: int, step: int) -> None:
        self.ratio = ratio
        self.step = step
        self.input_image_path = "./img/tmp/input/"
        self.input_image_paths = sorted(glob.glob(f"{self.input_image_path}*"))
        self.tile_image_path = "./img/tmp/tile/"
        self.tile_image_paths = sorted(glob.glob(f"{self.tile_image_path}*"))
    

    def _resize(self) -> None:
        '''
        tile画像の一要素のpx数に合わせてinput画像をリサイズし、元のディレクトリに格納し直す
        '''
        path = self.input_image_path
        for file in os.listdir(path):
            ext = file.split(".")[-1]
            if ext not in ("png", "jpg", "jpeg"):
                continue
            filepath = os.path.join(path, file)
            tmp_img = cv2.imread(filepath)
            tmp_img = cv2.resize(tmp_img, dsize=(0, 0), fx=self.ratio, fy=self.ratio)
            cv2.imwrite(filepath, tmp_img)


    def _create_layer_img(self) -> None:
        '''
        レイヤー画像を生成して重ね合わせる
        '''
        # 背景の白を透明化（内部関数）
        def _convert_to_transparent(filename: str) -> cv2.Mat:
            img = cv2.imread(filename)
            mask = np.all(img[:,:,:] == [255, 255, 255], axis=-1)
            dst = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            dst[mask,3] = 0
            return dst

        # 対応する画像を重ね合わせる（内部関数）
        def _overlap(base_img_path: str, over_img_path: str) -> None:
            baseImg = cv2.imread(base_img_path)
            overImg = cv2.imread(over_img_path, cv2.IMREAD_UNCHANGED)

            width, height = baseImg.shape[:2]
            baseImg[0:width, 0:height] = (
                baseImg[0:width, 0:height] * (1 - overImg[:, :, 3:] / 255) # type: ignore
                 + overImg[:, :, :3] * (overImg[:, :, 3:] / 255) # type: ignore
            )
            cv2.imwrite(over_img_path, baseImg)

        
        for i in range(self.step):
            input_img = cv2.imread(self.input_image_paths[i])
            tile_img = cv2.imread(self.tile_image_paths[i])
            masked_img = cv2.bitwise_or(input_img, tile_img)

            masked_img_path = f"./img/tmp/masked/masked_{i}.png"
            cv2.imwrite(masked_img_path, masked_img)

            # 背景を透明化
            layered_img = _convert_to_transparent(masked_img_path)
            layered_img_path = f"./img/tmp/layered/layered_{i}.png"
            cv2.imwrite(layered_img_path, layered_img)

            if i > 0:
                below_layer = f'./img/tmp/layered/layered_{i-1}.png'
                current_layer = f"./img/tmp/layered/layered_{i}.png"
                _overlap(below_layer, current_layer)

    
    def _output_img(self) -> None:
        '''
        最終Layer画像をoutputディレクトリに保存
        '''
        last_step = self.step - 1
        last_img_path = f"./img/tmp/layered/layered_{last_step}.png"
        output_img_path = "./img/output/output.png"
        cv2.imwrite(output_img_path, cv2.imread(last_img_path))

    
    def run(self) -> None:
        self._resize()
        self._create_layer_img()
        self._output_img()
