# bag of words
import os
import random

import cv2

dir = "/Users/rinatahmetov/code/work/kamal/app/assets/images"


def get_files_in_folder(dir_path):
    return [x for x in os.listdir(dir) if x.endswith('.jpg')]


def init_random_bool_vector(mass_length, prob):
    return [random.random() - prob > 0 for x in xrange(mass_length)]


def train_vocabulary(file_list, bool_list, keypoint_detector, d_extrator, voc_size):
    pass


cv2.BOWKMeansTrainer()
