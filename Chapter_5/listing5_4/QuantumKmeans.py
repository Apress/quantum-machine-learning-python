import cirq 
from swap_test import SwapTest
import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
print('cirq version',cirq.__version__)
print('pandas version',pd.__version__)
print('numpy version',np.__version__)
print('matplotlib version', matplotlib.__version__)




class QuantumKMeans:
    
    def __init__(self,data_csv,num_clusters,features,copies=1000,iters=100):
        self.data_csv = data_csv
        self.num_clusters = num_clusters 
        self.features = features
        self.copies = copies
        self.iters = iters


    def data_preprocess(self):
        df = pd.read_csv(self.data_csv)
        print(df.columns)
        df['theta'] = df.apply(lambda x: math.atan(x[self.features[1]]/x[self.features[0]]), axis=1)
        self.X = df.values[:,:2]
        self.row_norms = np.sqrt((self.X**2).sum(axis=1))
        self.X = self.X/self.row_norms[:, np.newaxis]
        self.X_q_theta = df.values[:,2]
        self.num_datapoints = self.X.shape[0]
        
    def distance(self,x,y):
        st = SwapTest(prepare_input_states=True,input_state_dim=2, measure=True,
                      copies=self.copies)
        st.build_circuit(input_1_transforms=[cirq.ry(x)],
                         input_2_transforms=[cirq.ry(y)])
        prob_0, _ = st.simulate()
        _distance_ = 1 - prob_0
        del st
        return _distance_
        
    def init_clusters(self):
        self.cluster_points = np.random.randint(self.num_datapoints,size=self.num_clusters)
        self.cluster_datapoints = self.X[self.cluster_points,:]
        self.cluster_theta = self.X_q_theta[self.cluster_points]
        self.clusters = np.zeros(len(self.X_q_theta))
        
    def assign_clusters(self):
        self.distance_matrix = np.zeros((self.num_datapoints, self.num_clusters))
        for i,x in enumerate(list(self.X_q_theta)):
            for j,y in enumerate(list(self.cluster_theta)):
                self.distance_matrix[i, j] = self.distance(x,y)
        self.clusters = np.argmin(self.distance_matrix,axis=1)
    
    def update_clusters(self):
        updated_cluster_datapoints = []
        updated_cluster_theta = []
        for k in range(self.num_clusters):

            centroid = np.mean(self.X[self.clusters == k],axis=0)
            centroid_theta = math.atan(centroid[1]/centroid[0])
            updated_cluster_datapoints.append(centroid)
            updated_cluster_theta.append(centroid_theta)
        
        self.cluster_datapoints = np.array(updated_cluster_datapoints)
        self.cluster_theta = np.array(updated_cluster_theta)
    
    def plot(self):
        fig = plt.figure(figsize=(8, 8))
        colors = ['red', 'green', 'blue', 'purple','yellow','black']
        plt.scatter(self.X[:,0],self.X[:,1],c=self.clusters,
                    cmap=matplotlib.colors.ListedColormap(colors[:self.num_clusters]))
        plt.savefig('Clusters.png')

    def run(self):
        self.data_preprocess()
        self.init_clusters()
        for  i in range(self.iters):
            self.assign_clusters()
            self.update_clusters()
        self.plot()


if __name__ == '__main__':
    data_csv = '/home/santanu/Downloads/DataForQComparison.csv'
    num_clusters = 4
    qkmeans = QuantumKMeans(data_csv=data_csv,num_clusters=num_clusters,
                            iters=10,features=['Annual Income_k$', 'Spending Score_1_to_100'])
    qkmeans.run()
