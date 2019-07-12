from __future__ import (
    division,
    print_function,
)

import argparse
import os
import cv2
import pandas as pd

# import miscellaneous modules
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv

import matplotlib.patches as mpatches
import selectivesearch
import multiprocessing as mp

import multiprocessing
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="image directory to run predictions on")
    parser.add_argument("-p", "--predictions", help="name and address of prediction file")
    args = parser.parse_args()
    return args

def text_detection(image,east,save='ss_anchors.csv',confidence=0.5,width=320,height=320,image_name="image"):
    from imutils.object_detection import non_max_suppression
    import numpy as np
    import argparse
    import time
    import cv2
    orig = image.copy()
    (H, W) = image.shape[:2]
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    layerNames = [
    	"feature_fusion/Conv_7/Sigmoid",
    	"feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < confidence:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    text_boxes=[]
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # print(image+','+str(startX)+','+str(startY)+','+str(endX-startX)+','+str(endY-startY)+',text'+'\n')
        # print(str(startX)+str(startY)+str(endX-startX)+str(endY-startY)+'text'+'\n')
        with open(save,'a') as file:
            file.write(image_name+','+str(startX)+','+str(startY)+','+str(endX-startX)+','+str(endY-startY)+',text'+'\n')
            file.close()

    # cv2.imshow("Text Detection", orig)
    # cv2.waitKey(0)
    return text_boxes,orig

def find_matches(image,dir=dir,save='ss_anchors.csv'):
    img = cv2.imread(os.path.join(dir,image))
    text_boxes,img_text=text_detection(image=img,east='../AL_algos/frozen_east_text_detection.pb',save=save,image_name=image)
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(img, scale=100, sigma=0.9, min_size=10)
    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

        for x,y,w,h in candidates:
            cv2.rectangle(img_text,(x,y),(x+w,y+h),(0,0,255),2,8,0)
            with open(save,'a') as file:
                file.write(image+','+str(x)+','+str(y)+','+str(w)+','+str(h)+',object'+'\n')

def parallel_process(dir,save,images):

    pool = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())
    with poolcontext(processes=mp.cpu_count()) as pool:
        results = pool.map(partial(find_matches, dir=dir,save=save), images)

def main():
    args=get_arguments()
    if args.predictions is None:
        args.predictions = 'ss_anchors.csv'

    #list of images
    image_list = os.listdir(args.directory)

    # create score file
    with open(args.predictions,'w') as file:
        file.write('filename,x,y,w,h,type'+'\n')

    parallel_process(dir=args.directory,save=args.predictions,images = image_list)

if __name__ == "__main__":
    main()
