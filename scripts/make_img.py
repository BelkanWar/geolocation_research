import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD, PCA

SIZE_A = 30
SIZE_B = 2
SIZE_C = 20
PCA_SWITCH = True


data = {
    'x': [random.randint(1,10) for i in range(SIZE_A)] + [random.randint(1,5) for i in range(SIZE_B)] + [0 for i in range(SIZE_C)],
    'y': [random.randint(1,10) for i in range(SIZE_A)] + [random.randint(1,5) for i in range(SIZE_B)] + [0 for i in range(SIZE_C)],
    'z': [0 for i in range(SIZE_A)] + [random.randint(1,5) for i in range(SIZE_B)] +  [random.randint(1,8) for i in range(SIZE_C)]
    }

pca_data = np.array([[data['x'][i], data['y'][i], data['z'][i]] for i in range(SIZE_A+SIZE_B+SIZE_C)])


decomposer = TruncatedSVD(n_components=2)
decomposer.fit(pca_data)
eigen_vector = decomposer.components_



print(pca_data)
print(decomposer.transform(pca_data))
print(eigen_vector)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data['x'], data['y'], data['z'])
ax.quiver([0,0], [0,0], [0,0], eigen_vector[0:2, 0]*10, eigen_vector[0:2, 1]*10, eigen_vector[0:2, 2]*10)
ax.set_xlabel('$X$', fontsize=20)
ax.set_ylabel('$Y$', fontsize=20)
ax.set_zlabel('$Z$', fontsize=20)
plt.show()