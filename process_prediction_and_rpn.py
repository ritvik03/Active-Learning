import argparse
import pandas as pd
from shapely.geometry import Polygon
import cv2
import os
import shutil
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="image directory", required=True)
    parser.add_argument("-p", "--predictions", help="name and address of prediction result file", required=True)
    parser.add_argument("-s", "--ssearch", help="name and address of selective search result file", required=True)
    # parser.add_argument("-c", "--classes", help="name and address of class file", required=True)
    # parser.add_argument("-te", "--test_anno", help="name and address of test annotations file", required=True)
    # parser.add_argument("-tr", "--train_anno", help="name and address of train annotataions file", required=True)
    parser.add_argument("-r", "--result", help="result file")
    # parser.add_argument("-draw", "--enable_draw",action='store_true',help="to enable image drawing")
    args = parser.parse_args()
    return args

def iou(a,b):
    polygon = Polygon([(a['x'], a['y']), (a['x']+a['w'], a['y']), (a['x']+a['w'], a['y']+a['h']), (a['x'], a['y']+a['h'])])
    other_polygon = Polygon([(b['x'], b['y']), (b['x']+b['w'], b['y']), (b['x']+b['w'], b['y']+b['h']), (b['x'], b['y']+b['h'])])
    intersection = polygon.intersection(other_polygon)
    iou = intersection.area/(polygon.area+other_polygon.area-intersection.area)
    return iou

def create_folder(name):
    if name in os.listdir('.'):
        shutil.rmtree(name)
    os.mkdir(name)

def drawfile(filename,pred_box,anchor_box,directory):
    if filename in os.listdir("temp_files/max_iou"):
        image = cv2.imread(os.path.join("temp_files/max_iou",filename))
    else:
        image = cv2.imread(os.path.join(directory,filename))
    cv2.rectangle(image,(int(pred_box[0]),int(pred_box[1])),(int(pred_box[0]+pred_box[2]),int(pred_box[1]+pred_box[3])),(0,255,0),2,8,0)
    cv2.rectangle(image,(int(anchor_box[0]),int(anchor_box[1])),(int(anchor_box[0]+anchor_box[2]),int(anchor_box[1]+anchor_box[3])),(0,0,255),2,8,0)
    cv2.imwrite(os.path.join("temp_files/max_iou",filename),image)

def find_Max(pred_entry,directory,result,df_rpn,draw=False):
    max_ious=[]
    filename = pred_entry['filename']
    file_anchors = df_rpn[df_rpn['filename']==filename]
    max_iou_value = 0
    iou_entry = pd.DataFrame()
    for index2, anchor_entry in file_anchors.iterrows():
        iou_calc=iou(pred_entry,anchor_entry)
        if iou_calc>max_iou_value:
            max_iou_value = iou_calc
            iou_entry = anchor_entry

    pred_box = [pred_entry['x'],pred_entry['y'],pred_entry['w'],pred_entry['h']]
    if not iou_entry.empty:
        iou_box = [iou_entry['x'],iou_entry['y'],iou_entry['w'],iou_entry['h']]
    else:
        iou_box=[0,0,0,0]
        iou_entry['x']=0
        iou_entry['y']=0
        iou_entry['w']=0
        iou_entry['h']=0
    max_ious.append({'filename':filename,'iou':max_iou_value,'pred_x':pred_entry['x'],'pred_y':pred_entry['y'],'pred_w':pred_entry['w'],'pred_h':pred_entry['h'],'anchor_x':iou_entry['x'],'anchor_y':iou_entry['y'],'anchor_w':iou_entry['w'],'anchor_h':iou_entry['h']})
    with open(result,'a') as f:
        f.write(filename+','+str(max_iou_value)+','+str(pred_entry['x'])+','+str(pred_entry['y'])+','+str(pred_entry['w'])+','+str(pred_entry['h'])+','+str(iou_entry['x'])+','+str(iou_entry['y'])+','+str(iou_entry['w'])+','+str(iou_entry['h'])+'\n')
    if draw:
        drawfile(filename,pred_box,iou_box,directory)

def parallel_process(result,dir,df_pred,df_rpn,draw):
    pool = mp.Pool(mp.cpu_count())
    with poolcontext(processes=mp.cpu_count()) as pool:
        results = pool.map(partial(find_Max, directory=dir,result=result,df_rpn=df_rpn,draw=draw), (pred_entry for index,pred_entry in tqdm(df_pred.iterrows())))
    # print(len(max_ious))
    # print(max_ious)


def main():
    args=get_arguments()

    # Folder for storing the result of this code
    create_folder('temp_files')

    # Read score files
    df_pred= pd.read_csv(args.predictions)
    df_rpn = pd.read_csv(args.ssearch)

    # If images to be drawn
    # if args.enable_draw:
    create_folder('temp_files/max_iou')

    # If result file not specified
    if args.result is None:
        args.result='temp_files/result.csv'

    max_ious=[]
    with open(args.result,'w') as fd:
        fd.write('filename,iou,pred_x,pred_y,pred_w,pred_h,anchor_x,anchor_y,anchor_w,anchor_h\n')

    parallel_process(result=args.result,dir=args.directory,df_pred=df_pred,df_rpn=df_rpn,draw=True)

if __name__ == '__main__':
    main()
