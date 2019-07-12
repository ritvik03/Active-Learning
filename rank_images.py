import pandas as pd
import numpy as np
import cv2
import shutil
import os
import matplotlib.pyplot as plt
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="image directory", required=True)
    parser.add_argument("-r", "--result", help="name and address of result file", required=True)
    parser.add_argument("-p", "--predictions", help="name and address of prediction file", required=True)
    parser.add_argument("-c", "--classes", help="name and address of class file", required=True)
    parser.add_argument("-te", "--test_anno", help="name and address of test annotations file", required=True)
    parser.add_argument("-tr", "--train_anno", help="name and address of train annotataions file", required=True)
    args = parser.parse_args()
    return args

def find_weights(test_anno,train_anno,classfile):
    df_train_anno = pd.read_csv(train_anno,header=None,names=['filename','x','y','w','h','class'])
    df_test_anno = pd.read_csv(test_anno,header=None,names=['filename','x','y','w','h','class'])
    df_train_test=df_train_anno.append(df_test_anno)
    df_train_test['filename'] = df_train_test['filename'].apply(os.path.basename)
    trained_images=list(set(df_train_test['filename']))
    df_train_test= df_train_test.drop(columns=['x','y','w','h']).rename(columns={'filename':'count'})
    group_train_test = df_train_test.groupby('class').count()
    group_train_test.to_csv('temp_files/train_test_count.csv')
    train_test_count = pd.read_csv('temp_files/train_test_count.csv')
    classes_file = (pd.read_csv(classfile,header=None))
    classes_file=classes_file.rename(columns={0:'class',1:'index'})
    classes_list = list(classes_file['class'])
    train_test_count=train_test_count[train_test_count['class'].isin(classes_list)]
    return train_test_count,trained_images

def weights_func(classname):
    df_freq = pd.read_csv('temp_files/train_test_count.csv')
    count=(df_freq.loc[df_freq['class'] == classname, 'count']).values[0]
    return (df_freq['count'].sum()/count)

def create_folder(name):
    if name in os.listdir('.'):
        shutil.rmtree(name)
    os.mkdir(name)

def saveimages(filtered_unified):
    create_folder('temp_files/ordered_result')
    for i,image in enumerate(list(filtered_unified['filename'])):
        shutil.copy2('temp_files/max_iou/'+image,'temp_files/ordered_result/'+str(i)+'_'+image)


def main():
    args=get_arguments()
    result=pd.read_csv(args.result)
    result=result[['filename','iou','pred_x','pred_y','pred_w','pred_h']]
    result = result.rename(columns={'pred_x':'x','pred_y':'y','pred_w':'w','pred_h':'h'})
    print(result)

    df_pred=pd.read_csv(args.predictions)

    zero_iou_1 = (result.loc[result['iou']==0])#.drop(columns=['anchor_h','anchor_w','anchor_y','anchor_x'])
    non_zero_iou_1=result.drop(result.index[result['iou']==0].tolist())#.drop(columns=['anchor_h','anchor_w','anchor_y','anchor_x'])
    zero_iou = pd.merge(zero_iou_1,df_pred,on=['x','y','w','h','filename']).sort_values('score',ascending=False)
    non_zero_iou = pd.merge(non_zero_iou_1,df_pred,on=['x','y','w','h','filename']).sort_values('iou',ascending=False)
    zero_iou.to_csv("temp_files/zero_iou.csv",index=False)
    non_zero_iou.to_csv("temp_files/non_zero_iou.csv",index=False)

    train_test_count,trained_images=find_weights(test_anno=args.test_anno,train_anno=args.train_anno,classfile=args.classes)

    agreement = non_zero_iou['score']*non_zero_iou['iou']
    non_zero_iou['disagreement'] = (1-agreement).where(non_zero_iou.score<0.7,other=1-non_zero_iou.score)
    non_zero_iou = non_zero_iou.sort_values('disagreement',ascending=False).drop(columns=['x','y','w','h'])
    non_zero_iou['disagreement']*=non_zero_iou['label_name'].apply(weights_func)#(classname=non_zero_iou['label_name'])
    non_zero_iou = non_zero_iou.sort_values('disagreement',ascending=False)

    agreement = zero_iou['score']
    zero_iou['disagreement'] = 1-agreement#.where(non_zero_iou.score<0.7,other=non_zero_iou.score)
    zero_iou['disagreement']*=zero_iou['label_name'].apply(weights_func)
    zero_iou = zero_iou.sort_values('disagreement',ascending=False).drop(columns=['x','y','w','h'])

    zero_iou.to_csv("temp_files/zero_iou.csv",index=False)
    non_zero_iou.to_csv("temp_files/non_zero_iou.csv",index=False)
    unified = zero_iou.append(non_zero_iou).drop_duplicates()

    grouped_non_zero = non_zero_iou.groupby(non_zero_iou['filename'])#.mean()#.add_prefix('mean_')#.mean()#.rename(columns={'filename':'filename','agreement':'mean_agreement'})#.sort_values('agreement')
    grouped_zero = zero_iou.groupby(non_zero_iou['filename'])
    grouped_unified = unified.groupby(non_zero_iou['filename'])

    non_zero_final = grouped_non_zero.aggregate(np.sum).sort_values('disagreement',ascending=True)
    zero_final = grouped_zero.aggregate(np.sum).sort_values('disagreement',ascending=True)
    unified_final = grouped_unified.aggregate(np.sum).sort_values('disagreement',ascending=True)

    non_zero_final.to_csv("temp_files/non_zero_iou.csv",index=True)
    zero_final.to_csv("temp_files/zero_iou.csv",index=True)
    unified_final.to_csv("temp_files/unified.csv",index=True)

    unified = pd.read_csv("temp_files/unified.csv")
    non_zero = pd.read_csv('temp_files/non_zero_iou.csv')
    zero = pd.read_csv('temp_files/zero_iou.csv')

    filtered_unified = unified[~unified['filename'].isin(trained_images)]
    filtered_zero = zero[~zero['filename'].isin(trained_images)]
    filtered_non_zero = non_zero[~non_zero['filename'].isin(trained_images)]
    filtered_non_zero.to_csv("temp_files/non_zero_iou.csv",index=False)
    filtered_zero.to_csv("temp_files/zero_iou.csv",index=False)
    filtered_unified.to_csv("temp_files/unified.csv",index=False)

    filtered_unified.to_csv("result.csv",index=False)
    filtered_unified = filtered_unified.sort_values('disagreement',ascending=False)
    filtered_unified.to_csv('result_for_AL_multilabelImg.csv')

    saveimages(filtered_unified)


if __name__ == "__main__":
    main()
