import sys
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
X=np.genfromtxt('TempHDBSCANfile_9.txt')
Y=np.genfromtxt('TempHDBSCANparameters_9.txt')
MinClusterSize=int(Y[0])
MinSamples=int(Y[1])

clusterer = hdbscan.HDBSCAN(min_cluster_size=MinClusterSize,min_samples=MinSamples,
                            cluster_selection_method='eom',approx_min_span_tree=False,core_dist_n_jobs=1)
clusterer.fit(X)
tree=clusterer.condensed_tree_.plot(select_clusters=True,selection_palette=sns.color_palette())
np.savetxt('HDBSCANoutputs_9\Labels.txt',clusterer.labels_, fmt='%d',comments='')
np.savetxt('HDBSCANoutputs_9\Persistence.txt',clusterer.cluster_persistence_,comments='')
np.savetxt('HDBSCANoutputs_9\Probabilities.txt',clusterer.probabilities_,comments='')    