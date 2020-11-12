import os
import glob 

list_path = []

train = open('filelists/train.txt', 'a')
test = open('filelists/test.txt', 'a')

for path in glob.glob("data/doanngocle/*"):
    list_path.append(path)

for i in range(0,50):
    test.writelines(list_path[i]+"\n")

for i in range(51,len(list_path)):
    train.writelines(list_path[i]+"\n")

test.close()
train.close()
