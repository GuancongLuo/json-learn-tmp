import csv

import os
import pathlib
import time
current_time=time.time()


f = open("airsim.txt",'r+')               # 返回一个文件对象 
f2 = open("airsim2.txt",'r+')               # 返回一个文件对象 

line = f.readline()               # 调用文件的 readline()方法 
while line: 
    print(line)                  # 后面跟 ',' 将忽略换行符 
    #print(line, end = '')　      # 在 Python 3 中使用 
    line = f.readline() 
    new_line =str(current_time) + line
    f2.write(new_line)

f.close() 
