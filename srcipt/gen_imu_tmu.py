import csv

import os
import pathlib

with open('test_20220112191631_imu_pose.csv','a+',newline='') as csvfile:
    spamreader = csv.reader(csvfile,delimiter=' ', quotechar='|')
    for row in spamreader:
        print(' '.join(row))