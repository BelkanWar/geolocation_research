import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../src/")
import utils
import matplotlib.pyplot as plt
import seaborn as sns

timestamp_list = []

for root, folder, files in os.walk("../splitted_data/"):
    for file in files:
        data = utils.data_parsing(utils.read_raw_data(os.path.join(root, file)))
        
        for timestamp in data['start_time']:
            timestamp_list.append(timestamp)
print('plotting')
sns.displot(data={'timestamp':timestamp_list}, x='timestamp', kind='kde')
plt.show()