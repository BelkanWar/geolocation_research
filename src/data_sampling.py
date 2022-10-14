import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import csv

TARGET_PATH = "../data/small_jakarta_sample.csv"
SOURCE_PATH = "../data/jakarta_sample.csv"

if os.path.exists(TARGET_PATH):
    os.remove(TARGET_PATH)

imsi_count_dict = {}

with open(SOURCE_PATH) as f_read:
    for i in csv.reader(f_read, delimiter='|'):
        imsi = i[0]
        if imsi in imsi_count_dict:
            imsi_count_dict[imsi] += 1
        else:
            imsi_count_dict[imsi] = 1

imsi_count_list = [[imsi, imsi_count_dict[imsi]] for imsi in imsi_count_dict]
imsi_count_list.sort(key=lambda x:x[1], reverse=True)

imsi_set = set([i[0] for i in imsi_count_list[:1000]])

f_read = open(SOURCE_PATH)
f_write = open(TARGET_PATH, 'a', newline='')

w = csv.writer(f_write, delimiter='|')

for i in csv.reader(f_read, delimiter='|'):
    if i[0] in imsi_set:
        w.writerow(i)

f_read.close()
f_write.close()