import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np

# Parameters
center_hight_1 = -2
center_hight_2 = 2
sigma = 3
SAMPLE_SIZE = 50
BOOTSTRAP_SIZE = 5000

def bootstrapping(data):
    mean_list = []
    for i in range(BOOTSTRAP_SIZE):
        mean_list.append(np.mean(random.choices(data, k=len(data))))
    return mean_list

A = [random.normalvariate(mu=center_hight_1, sigma=sigma) for i in range(SAMPLE_SIZE)]
B = [random.normalvariate(mu=center_hight_2, sigma=sigma) for i in range(SAMPLE_SIZE)]

data = {'value':A+B, 'group':['A']*SAMPLE_SIZE + ['B']*SAMPLE_SIZE}
mean = {'mean':bootstrapping(A) + bootstrapping(B), 'group':['A']*BOOTSTRAP_SIZE + ['B']*BOOTSTRAP_SIZE}

fig, axs = plt.subplots(2, 1, sharex=True)
sns.kdeplot(data=data, x='value', hue='group', ax=axs[0])
sns.kdeplot(data=mean, x='mean', hue='group', ax=axs[1])

plt.show()