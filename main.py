import cv2
import numpy as np
import random

from src.utils import *
from src.image_preparer import *
from src.tile_generator import *
from src.image_drawer import *

SRC_IMAGE = "./img/source/dog.png"

if __name__ == "__main__":
    src = SRC_IMAGE
    src_image = cv2.imread(src, -1)
    step = 20
    
    # ソース画像の前処理
    print("=== Preparing Source Image ===")
    image_preparer = ImagePreparer(src_image=src_image, step=step)
    image_preparer.run()

    # タイル画像の生成
    print("=== Creating Tile Image ===")
    tile_src = "./img/source/lambda.png"
    tile_image = cv2.imread(tile_src)
    x, y, _ = src_image.shape
    tile_generator = TileGenerator(tile_img=tile_image, step=step, src_x=x, src_y=y)
    tile_generator.run()

    # 出力画像の生成
    print("=== Generating Output Image ===")
    ratio = tile_generator.get_tile_size()
    image_drawer = ImageDrawer(ratio=ratio, step=step)
    image_drawer.run()    
