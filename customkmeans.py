import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([
[1,2],
[1.5,1.8],
[5,8],
[8,8],
[1,0.6],
[9,11]
]
)

#plt.scatter(X[:,0],X[:,1], s= 150, linewidths=5)
#plt.show()

colors = 10*["g.","r.","c.","b.","k."]


class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def fit(self,data):
        pass
    def predict(self,data):
        pass
