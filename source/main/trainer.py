# bag of words
import os
import random

import cv2

folder = "/Users/rinatahmetov/Downloads/101_ObjectCategories"


def get_files_in_folder(dir_path):
    training_names = os.listdir(dir_path)
    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        folder = os.path.join(dir_path, training_name)
        class_path = [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith('.jpg')]
        image_paths += class_path
        image_classes += [class_id] * len(class_path)  # ?????
        class_id += 1
    return image_paths


def init_random_bool_vector(mass_length, prob):
    return [random.random() - prob > 0 for _ in xrange(mass_length)]


def get_set(image_paths, bool_vector, train=0):
    return [image_paths[x] for x in xrange(len(image_paths)) if bool_vector[x] == train]


def train_vocabulary(file_list, voc_size, keypoint_detector="SIFT", d_extrator="SIFT"):
    bow_mt = cv2.BOWKMeansTrainer(voc_size)
    fea_det = cv2.FeatureDetector_create(keypoint_detector)
    des_ext = cv2.DescriptorExtractor_create(d_extrator)
    for x in file_list:
        image = cv2.imread(x)
        features = fea_det.detect(image)
        _, des = des_ext.compute(image, features)
        bow_mt.add(image, des)
    return bow_mt


def extract_features_from_image(detector, bow_extractor, img):
    image = cv2.imread(img)
    features = detector.detect(image)
    _, desc = bow_extractor.compute(features)
    return desc


def train_classifier(train_data, train_responses):
    classifier = cv2.RTrees()
    rtree_params = dict(max_depth=11, min_sample_count=5,
                        use_surrogates=False, max_categories=15,
                        calc_var_importance=False, nactive_vars=0,
                        max_num_of_trees_in_the_forest=200,
                        term_crit=(cv2.TERM_CRITERIA_MAX_ITER, 1000, 1)
                        )
    return classifier.train(
        train_data, cv2.CV_ROW_SAMPLE,
        train_responses, params=rtree_params)


images = get_files_in_folder(folder)
bool_vec = init_random_bool_vector(len(images), 0.5)
print bool_vec
train_set = get_set(images, bool_vec)
print train_set
