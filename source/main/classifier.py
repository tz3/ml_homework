import os

import cv2
from numpy.random import random


class BOWClassifier(object):
    def __init__(self, detector_type, descriptor_type, voc_size):
        self.detector = cv2.FeatureDetector_create(10, True, detector_type)
        self.descriptor_extractor = cv2.DescriptorExtractor_create(descriptor_type)
        self.voc_size = voc_size
        self.classifier = None
        self.vocabulary = None
        self.bow_detector = cv2.BOWImgDescriptorExtractor(
            self.descriptor_extractor, cv2.DescriptorMatcher_create(descriptor_type)
        )
        self.classes = []

    def classify(self, img):
        pass

    def train_vocabulary(self, images, is_voc):
        train_imgs = get_set(images, is_voc, True)
        trainer = cv2.BOWKMeansTrainer(self.voc_size)
        for img in train_imgs:
            matrix = cv2.imread(img)
            features = self.detector.detect(matrix)
            _, des = self.descriptor_extractor.compute(matrix, features)
            trainer.add(des)
        trainer.cluster()
        self.bow_detector.setVocabulary(trainer.cluster())

    def train(self, images, classes, is_train):
        self.train_vocabulary(images, is_train)
        data, response = self.extract_train_data(images, is_train, classes)
        self.train_rtree(data, response)

    def extract_train_data(self, images, is_train, responses):
        train_images = get_set(images, is_train, True)
        train_data = []
        for i, img in enumerate(train_images):
            image = cv2.imread(img)
            features = self.detector.detect(image)
            _, des = self.bow_detector.compute(features)
            train_data[i] = des
        return train_data, get_set(responses, is_train, True)

    def train_rtree(self, train_data, train_responses):
        rtree = cv2.RTrees()
        rtree_params = dict(max_depth=11, min_sample_count=5,
                            use_surrogates=False, max_categories=15,
                            calc_var_importance=False, nactive_vars=0,
                            max_num_of_trees_in_the_forest=200,
                            term_crit=(cv2.TERM_CRITERIA_MAX_ITER, 1000, 1)
                            )
        rtree.train(
            train_data, cv2.CV_ROW_SAMPLE,
            train_responses, params=rtree_params
        )
        self.classifier = rtree

    def predict(self, img):
        image = cv2.imread(img)
        features = self.detector.detect(image)
        _, desc = self.bow_detector.detect(image, features)
        return self.classifier.predict(image)

    def extract_features_from_image(self, img):
        image = cv2.imread(img)
        _, desc = self.bow_detector.compute(image, self.detector.detect(image))
        return desc


def init_random_bool_vector(mass_length, prob):
    return [random.random() - prob > 0 for _ in xrange(mass_length)]


def get_set(image_paths, bool_vector, train=0):
    return [image_paths[x] for x in xrange(len(image_paths)) if bool_vector[x] == train]


def predict_on_test_data(bow_classifier, images, is_train):
    test_data = get_set(images, is_train, False)
    return map(bow_classifier.predict, test_data)


def test_responses(responses, is_train):
    return get_set(responses, is_train, False)


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
    return image_paths, image_classes


def calculate_missclassification(responses, predictions, total):
    matched = 0
    for r, p in zip(responses, predictions):
        if r == p:
            matched += 1
    return matched / total


if __name__ == '__main__':
    classifier = BOWClassifier("SIFT", "SIFT", 25)
    files, classes = get_files_in_folder("/home/blvp/Downloads/101_ObjectCategories")
    is_train_random = init_random_bool_vector(len(files), 0.5)
    classifier.train(files, classes, is_train_random)
    predictions = predict_on_test_data(classifier, files, is_train_random)
    responses_test = get_set(classes, is_train_random, False)
    result = calculate_missclassification(responses_test, predictions, len(files))
    print result
