import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x,y = make_blobs(n_samples =100, centers = 6, random_state=0,cluster_std=2)


estimator = KMeans(6)

estimator.fit(x)

y_means = estimator.predict(x)

plt.scatter(x[:,0],x[:1],s=50)
plt.scatter(x[:,0],x[:,1],c=y_means, s=50, cmap  ='rainbow')
plt.show()
