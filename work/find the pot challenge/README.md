## “Find the pot” Challenge
The goal of the Find the pot challenge is to locate the pot(s) in which poinsettia are planted in each
image. 

### Data set:
Use the **side view data set** for this challenge.

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