from typing import Any
import cv2
import numpy as np

from .utils import *


class TileGenerator:
    def __init__(self, tile_img: cv2.Mat, step: int, src_x: int, src_y: int) -> None:
        self.tile_img = tile_img
        self.step = step
        self.src_x = src_x
        self.src_y = src_y
    
    
    def _prepare_tile(self) -> None:
        '''
        タイル画像の生成
        '''
        # リサイズ
        self.tile_img = cv2.resize(thumnail(self.tile_img, 10), dsize=(0, 0), fx=0.5, fy=0.5)  # type: ignore

        # ソース画像サイズの空行列を作成
        tile = np.zeros((self.src_x, self.src_y)).tolist()

        # 空行列のそれぞれのセルにタイル画像を埋める
        for i in range(self.src_x):
            for j in range(self.src_y):
                tile[i][j] = self.tile_img

        # タイル画像を結合し、一枚の大きなタイルを生成
        self.base_tile = cv2.vconcat([cv2.hconcat(tile_h) for tile_h in tile])  # type: ignore


    def _adjust_brightness(self):
        '''
        タイル画像を濃度別に生成
        '''
        def _adjust(img: Any, alpha: float=1.0, beta: float=0.0):
            # 積和演算を行う。
            dst = alpha * img + beta
            # [0, 255] でクリップし、uint8 型にする。
            return np.clip(dst, 0, 255).astype(np.uint8)
        
        for i in range(self.step):
            tone = i * 10
            # showImage(adjust(base_tile, alpha=0.5, beta=tone))
            adjust_img = _adjust(self.base_tile, alpha=0.5, beta=tone)
            cv2.imwrite(f'./img/tmp/tile/tile_{i}.png', adjust_img)


    def get_tile_size(self) -> int:
        '''
        タイル画像の１要素のサイズを返却
        '''
        return self.tile_img.shape[0]

    
    def run(self) -> None:
        self._prepare_tile()
        self._adjust_brightness()
