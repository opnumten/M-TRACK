# M-TRACK

M-TRACK is a live cell imaging analysis tool. 

The package is written in Python 3.7.

requirement:
wx
numpy
matplotlib
sqlite3
argparse
pandas
sklearn
Pillow
mahotas
scikit-image



The input file include:
fluorescence images
corresponding segmented cell mask images
selected segmented cell mask image for calculation of mean cell contour
database file (sqlite file) of cell tracking results (cellprofiler)

we share some data  for test in the following link:
https://drive.google.com/drive/folders/1KwVwR9moblQlA0U2aDPWHNeKDNONrk9J

Run MTRACK_GUI to start processing, select input folder and output folder

1 in morphology analysis, ptsnum is the number of points of cell outputline. This value depends on the resolution of images and requirements for analysis. Default is 150.

2 in Haralick analysis, fluor interval is the time step of fluorescence images depends on the experiment setting. Default is 1.

3 in trajectory analysis, minmum traj length is the lower threshold of trajectory length you want to analyze. Default is 12.  After extraction of single cell trajectory and scaling analysis, choose an trajectory to plot from the folder.

