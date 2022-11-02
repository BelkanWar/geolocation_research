import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../src/")
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from scipy import stats

# Parameters
center_hight_1 = 5
center_hight_2 = 6
sigma = 3
SAMPLE_SIZE = 100
PLOT_PDF = True

def bootstrapping(data):
    mean_list = []
    for i in range(5000):
        mean_list.append(np.mean(random.choices(data, k=len(data))))
    return mean_list


data1, data2 = [{'x':[], 'y':[]} for i in range(2)]

if os.path.exitsts("../temp/"):
    os.mkdir("../temp/")

for root, folder, files in os.walk("../temp/"):
    for file in files:
        os.remove(os.path.join(root, file))

for i in range(SAMPLE_SIZE):
    fig, axs = plt.subplots(1,4, sharey=True, gridspec_kw=dict(width_ratios=[2,5,5,2]))
    for idx in [1,2]:
        axs[idx].set_xlim([-10, 10])
        axs[idx].set_ylim([-2.5, 15])

    data1['x'].append(random.normalvariate(mu=0, sigma=sigma))
    data1['y'].append(random.normalvariate(mu=center_hight_1, sigma=sigma))

    data2['x'].append(random.normalvariate(mu=0, sigma=sigma))
    data2['y'].append(random.normalvariate(mu=center_hight_2, sigma=sigma))

    axs[1].scatter(data1['x'], data1['y'], c='r')
    axs[2].scatter(data2['x'], data2['y'], c='b')

    fig.savefig(f"../temp/{i}.png")


    if PLOT_PDF and i > 9:
        mean_dist1 = bootstrapping(data1['y'])
        mean_dist2 = bootstrapping(data2['y'])
        sns.kdeplot(data={'y':mean_dist1}, y='y', ax=axs[0])
        sns.kdeplot(data={'y':mean_dist2}, y='y', ax=axs[3])

        fig.savefig(f"../temp/{i}_se.png")
    
    plt.close()
    
    


