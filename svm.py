import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Mahcine:
    def __init__(self, visualization = True):
            self.visualization = visualization
            self.colors = {1:'r', -1:'b'}
            if(self.visualization):
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(1,1,1)
    #train
    def fit(self, data):
        self.data = data

        opt_dict = {}
        transforms = [[1,1],[-1,-1],[-1,1],[1,-1]]

        all_data = []
        for yi in self.data:
            for featuresset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        step_sizes = [self.max_feature_value * 0.1,
        self.max_feature_value * 0.01,
        # Point of expense:
        self.max_feature_value * 0.001]
        # Expensive , b doesn't need to take same steps as ||w||
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # Convex
            optimized = False
            while not optimized:
                pass

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification

data_dict = {'-1':np.array([[1,7],[2,8],[3,8]]),'1': np.array([[5,1],[6,-1],[7,3]])}
