# <----------------------> import <------------------>
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import *

# <----------------------> internal import <---------->
from DIPlib.intensityTransform import *
from DIPlib.enhancements import *
from DIPlib.fourier import *
from DIPlib.filters.frequency import *
from DIPlib.segmentations import *
from DIPlib.morphology import *
from DIPlib.features.regions import *

from skimage.exposure import equalize_hist
import skimage.morphology as skmorph

# <-----------------------> main script <------------->

DATABASE_PATH = "input/Leaves/"

if __name__ == "__main__":
    # "input/Leaves/1/leaf_01.jpg"
    input_files_1 = glob(DATABASE_PATH + "1/" + "*")
    input_files_2 = glob(DATABASE_PATH + "2/" + "*")
    input_files = input_files_1 + input_files_2
    print(input_files)

    for f in input_files:
        # Read image
        input_img = cv.imread(f)
        rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)




        # - Color
        gb_diff = rgb_img[:,:,1].astype(float) - rgb_img[:,:,2].astype(float)
        gb_diff = np.clip(gb_diff, 0, 255).astype(np.uint8)
        _, seg_img = cv.threshold(gb_diff, None, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # - Morphological
        stre = skmorph.disk(9)
        morph_img = removeFragments(seg_img, thresh_ratio=0.05)
        morph_img = fillHoles(morph_img)
        morph_img = cv.morphologyEx(morph_img, cv.MORPH_CLOSE, stre)
        morph_img = fillHoles(morph_img)

        # - Feature Extraction
        _, eccen = regionBasedFeatures(morph_img, "eccentricity")
        # print(eccen[0])

        # - Classification Tree
        if eccen[0] < 0.8:
            leaf_class = "1"
        else:
            leaf_class = "2"

        print(leaf_class)
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_img)
        # plt.subplot(1, 2, 2)
        # plt.title(f"Leaf Class: {leaf_class}")
        # plt.imshow(morph_img, cmap="gray")
        # plt.show()
