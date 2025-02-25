## “Find the colours” Challenge
The goal of the Find the colours challenge is to find the red and green leaves in the plant and
compute the ratio of red to green. This could be used to estimate plant maturity, one of the criteria
for grading poinsettia plants used in nurseries. This is a challenge that could be completed using
standard image processing methods for colour segmentation or machine learning methods.

### Data set:
Use the **sea of plants** data set for this challenge.

### Result format 
Entries should indicate the bounding box of the pot. Results should be provided in a csv file, with
5 columns:
image file name

 *pot.x top left pot.y top left pot.x bottom right pot.y bottom right*

For each plant pot you have located in the image, list the name of the test image file followed
by the (x, y) coordinates of the corners of the bonding box outlining the plant pot. Use the pixel
coordinates in the image (where (0, 0) is the upper left corner of the image). If you find multiple
plant pots in an image, then insert multiple rows in the csv file, one per plant pot.

### Scoring
Entries will be scored using the IoU metric, comparing the overlap of the entry
bounding boxes with our ground truth for each of the images in the evaluation data set. The highest average IoU score wins (averaged over all the images in the evaluation data set).