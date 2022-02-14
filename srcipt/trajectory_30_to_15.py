import csv

import os
import pathlib
import time
from typing import Counter
current_time=time.time()


f = open("Trajectory.txt",'r+')               # 返回一个文件对象 
f2 = open("Trajectory15.txt",'r+')               # 返回一个文件对象 
cout = 1
line = f.readline()               # 调用文件的 readline()方法 
while line: 
    print(line)                  # 后面跟 ',' 将忽略换行符 
    #print(line, end = '')　      # 在 Python 3 中使用 
    line = f.readline() 
    if cout % 2 == 0:
        print("ok")
        f2.write(line)
    cout =cout+1
    print(cout)

f.close() 
