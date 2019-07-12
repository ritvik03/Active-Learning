import argparse
import keras
import os
import pandas as pd

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from tqdm import tqdm
import csv

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="inference model to be used for predictions")
    parser.add_argument("-d", "--directory", help="image directory to run predictions on")
    parser.add_argument("-p", "--predictions", help="name and address of prediction file")
    parser.add_argument("-c", "--classes", help="classes file")
    args = parser.parse_args()
    return args

def load_classes(classfile):
    LABELS = open(classfile).read().strip().split("\n")
    classes_retrain = list(L.split(",")[0] for L in LABELS)
    indexes = list(range(len(classes_retrain)))
    labels_to_names = dict(zip(indexes, classes_retrain))
    return labels_to_names

def create_score_list(model,filenames,image_dir,save,labels_to_names):
    # filewise_uncertainity=[]
    # scores_list=[]
    csv_columns = ['filename','score','label_name','x','y','w','h']
    with open(save,'w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

    for file in tqdm(filenames):
        image = read_image_bgr(image_dir+file)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0:
                break
            data = {'filename': file,'score': score,'label_name':labels_to_names[label],'x': box[0],'y': box[1],'w': box[2]-box[0],'h':box[3]-box[1]}
            with open(save,'a') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writerow(data)

            f.close()


def main():
    args=get_arguments()
    if args.predictions is None:
        args.predictions = 'predictions.csv'
    # print(args.model)
    # print(args.directory)
    # print(args.predictions)
    # print(args.classes)

    #loading model
    model = models.load_model(args.model, backbone_name='resnet50')
    #ADD THIS LINE TO CONVERT TO INFERENCE MODEL
    model = models.convert_model(model)

    #loading class names
    labels_to_names=load_classes(args.classes)
    #loading filenames
    filenames = os.listdir(args.directory)

    create_score_list(model=model,filenames=filenames,image_dir=args.directory,save=args.predictions,labels_to_names=labels_to_names)



if __name__ == "__main__":
    main()
