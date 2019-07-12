# Application of ACTIVE LEARNING in object detection
What is active learning?
Active learning is a form of learning in which teaching strives to involve students in the learning process more directly than in other methods.

What has been done?
It is a process of ranking images on basis of _correctiveness_ of their predictions. This in turn serves 2 purpose:
1. To filter images with high correctness of prediction and consider their predictions as their true label. This saves  a lot of labelling time for classes with good accuracy. This also brings down the sample space of unknown data labels.
2. To create a list of images with low correctness of prediction (classification or regression) to be queried and annotated by an oracle. This reduces the standard deviation of model's prediction thereby imporving the mAP score at a higher rate than that with random data sampling.

##Repository Structure:

###Documentation/
├── ActiveLearning_MultiLabelImg
├── created_files
│   ├── predictions.csv
│   └── ss.csv
├── create_prediction_scores.py
├── create_selectivesearch_scores.py
├── papers
├── process_prediction_and_rpn.py
├── rank_images.py
├── README.md
├── requirements.txt
├── results
│   ├── failed trials
│   │   ├── AL-exp-0.5-pool-1
│   │   └── AL-log-0.5-pool-1
│   ├── filewise 20_percent selected over 20_percent seed.png
│   ├── positive result
│   │   └── filewise-seleted training 0.2-0.2
│   ├── random 20_percent selection over 20_percent seed.png
│   ├── random_train_over_seed_0.2-0.2
│   └── Seed training
└── temp_files
    ├── max_iou
    ├── non_zero_iou.csv
    ├── ordered_result
    ├── result.csv
    ├── train_test_count.csv
    ├── unified.csv
    └── zero_iou.csv
## USING THE REPOSITORY
###STEP 0: requirements.txt
1. Create a virtual environment
2. $ pip install -r requirements.txt

###STEP 1: create_prediction_scores.py
Use: This file generates prediction over the given dataset using retinanet model. Any other model can be used to create a similar prediction file of similar format. The file format must be a csv file with columns: 'filename','score','label_name','x','y','w','h'

USAGE: create_prediction_scores.py [-h] [-m MODEL] [-d DIRECTORY]
                                   [-p PREDICTIONS] [-c CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        inference model to be used for predictions
  -d DIRECTORY, --directory DIRECTORY
                        image directory to run predictions on
  -p PREDICTIONS, --predictions PREDICTIONS
                        name and address of prediction file
  -c CLASSES, --classes CLASSES
                        classes file


EXAMPLE: $ python create_prediction_scores.py -m ../../snapshots/seed_20.h5 -c ../retinanet_classes.csv -d ../images/

###STEP 2: create_selectivesearch_scores.py
Use: This file creates appropriate region proposal on each image of the dataset to mimic the RPN working inside the object detection framework. Any other Region Proposal Network (RPN) can be used in place of currently used selectivesearch. Given that this RPN is not trained on certain dataset.

USAGE: $ python create_selectivesearch_scores.py [-h] [-d DIRECTORY] [-p PREDICTIONS]

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        image directory to run predictions on
  -p PREDICTIONS, --predictions PREDICTIONS
                        name and address of prediction file

EXAMPLE: $ python create_prediction_scores.py -d ../images/ -p anchors.csv

###STEP 3: process_prediction_and_rpn.py
Use: This file generates a result file by combining both prediction and anchors and computes Localization Tightness score from regression of this predicted anchor boxes from RPN to the prediction.

USAGE: process_prediction_and_rpn.py [-h] -d DIRECTORY -p PREDICTIONS -s
                                     SSEARCH [-r RESULT]

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        image directory
  -p PREDICTIONS, --predictions PREDICTIONS
                        name and address of prediction result file
  -s SSEARCH, --ssearch SSEARCH
                        name and address of selective search result file
  -r RESULT, --result RESULT
                        result file

EXAMPLE: $ python process_prediction_and_rpn.py -d ../images -p created_files/predictions.csv -s created_files/selectivesearch.csv

###STEP 4: rank_images.py
Use: This file could have been part of process_prediction_and_rpn.py but I creates it for quick use and testing if result.csv (it's result file) is already available. It combines the query based on prediction score and localization tightness score to create a disagreement index of the image. This disagreement index shows how reliable are the predictions made. Higher disagreement images must be human annotated and lower ones could be reviewed to be directly added as labelled data.
Final result from these steps is saved in **temp_files/unified.csv**

USAGE: rank_images.py [-h] -d DIRECTORY -r RESULT -p PREDICTIONS -c CLASSES
                      -te TEST_ANNO -tr TRAIN_ANNO

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        image directory
  -r RESULT, --result RESULT
                        name and address of result file
  -p PREDICTIONS, --predictions PREDICTIONS
                        name and address of prediction file
  -c CLASSES, --classes CLASSES
                        name and address of class file
  -te TEST_ANNO, --test_anno TEST_ANNO
                        name and address of test annotations file
  -tr TRAIN_ANNO, --train_anno TRAIN_ANNO
                        name and address of train annotataions file

EXAMPLE: $ python rank_images.py -d ../images/ -r temp_files/result.csv -c ../retinanet_classes.csv -te ../retinanet_test.csv -tr ../retinanet_train.csv -p ../AL_algos/prediction_scores_box_best.csv

###papers
This directory contains all the papers reviewed and used in creating this repository. Some of the modification of query strategy are unique as they have been developed by testing them on dataset. Rest everything done comes from the papers.

###results
The approach followed was using 20% data as seed and querying another 20% to see the progress.
This directory has 3 parts:
1. Positive Results:
- contains the result which gave desired result. This was using _only classification score_ rather than both classification and regression scores for query strategy.
2. Failed Attempts:
- they are kept so as to keep track of approaches which were discussed in papers but aren't that affective as mentioned on our dataset.
3. mAP images : 2 different images:
- random query 20% over 20% seed
- filewise logarithmic decay 20% over 20% seed.
These images of tensorboard mAPs distinctively shows the later one to highly exceed the previous one. Both absolute value of mAP and the time(epoch) to reach that is significantly better in filewise logarithmic decay algorithm.

###temp_files
This directory is created to score intermediate csv files which are to be used in later approaches. Final result is also saved as unified.csv file in this folder.

####[NOTE]: Parallel processing is used wherever necessary still on a 12 core machine, to process ~21k images, all these steps take near to 2 days with maximum ram usage close to 4GB.


#ActiveLearning_MultilabelImg
##STEP 5: Labelling Queried Data
A module to integrate application of active learning on object detection with labelImg, a widely used labelling tool to annotate data. MultilabelImg has was created from basecode of labelImg. Active Learning MultilabelImg is the integration of active learning module with MultilabelImg. Apart for just active learning, this module could also be used to with some different ranking method.

INPUTS: image directory, class files directory, annotation directory, [optional] active learning ranking list (result of step 4).
USAGE: All the available options are integrated as GUI features so ease off the process for labelling team. One has to just select image directory after running the code using the command line:

**$ python ActiveLearning_MultilabelImg.py**

If an ranking for given image directory already present, it uses the same. Else, it asks the user if he/she wants to add ranking.csv for image ordering. If user does not selects the ranking file, it works just like normal multilabelImg.
Ranking.csv is in decreasing order of disagreement coefficient

#####In case of any doubts please contact:
######Ritvik Pandey
######ritvik.pandey@course5i.com
