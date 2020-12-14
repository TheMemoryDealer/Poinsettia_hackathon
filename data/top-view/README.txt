The poinsettia_dataset folder contains two subfolders; train and test. The train
folder contains the images that we will use for training and a file
"poinsettia_train.json" which contains the bounding box labels in coco format.
The coco format will be the official format in this challenge. The test folder
contain the images that we will use to evaluate your poinsettia detectors.

For this challenge you will need to upload a .json file with the prediction of
your detector in coco format. Then, we will evaluate your file with the ground
truth information. We will use the Average Precision (AP) (AP at IoU=.50:.05:.95)
metric to evaluate your detector. So, you should focus on this metric. For more
information about this metric please check the attached tutorial and the COCO
website:  https://cocodataset.org/#detection-eval


The file "results_sample.json" is an example of a submission file for this
challenge.
