import optparse
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from sklearn.cluster import KMeans
import itertools
from math import log

# to calculate the variation
def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)

def clustering(dataset,clusters_input,pca_components):
    # open the input dataset and read the contents of it
    df = open(dataset,'r')
    df_lines = df.readlines()

    # get the clusters and convert them into a tuple
    clusters_list = clusters_input.split(',')
    clusters_list = map(int, clusters_list)
    ks = tuple(clusters_list)

    # compute the number of supspaces
    nrOfSubspaces = len(ks)

    # get the features and labels (cluster centers lables)
    features = {}
    labels = {}

    # also store the indices of data points which belong to the labels
    true_clusters_indices = {}
    for r in range(0,ks[0]):
        true_clusters_indices[r] = []

    for i in range(0, len(df_lines)):
        row = df_lines[i].strip()
        row_list = row.split(';')
        labels[i] = row_list[0:ks[0]]
        for label in map(int,labels[i]):
            true_clusters_indices[label].append(i)
        features[i] = map(float, row_list[ks[0]:len(row_list)])

    # converting features into a np array
    features_values = features.values()
    features_array = np.array(features_values)

    # Applying PCA
    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features_array)

    # calculating the m dimensionalities
    m = features_pca.shape[1]/nrOfSubspaces

    # generate mapping Pj to the features
    firstDim = 0
    ranges = {}
    for k in range(0, nrOfSubspaces):
        if (k == nrOfSubspaces - 1):
            endOfRange = features_pca.shape[1]
        else:
            endOfRange = firstDim + m

        ranges[k] = [firstDim, endOfRange]
        firstDim = firstDim + m

    predicted_clusters = {} # to store the cluster labels for all the subspaces
    for subspace in range(0,nrOfSubspaces):
        print "Subspace:" + str(subspace)
        nrOfclusters = ks[subspace] # no of clusters per subspace

        # get the initial means or cluster centers based on input data
        kmeans = KMeans(n_clusters=nrOfclusters, init="k-means++").fit(features_pca)
        cluster_centers = kmeans.cluster_centers_

        range_value = ranges[subspace]
        cluster_centers = cluster_centers[:, range_value[0]:range_value[1]]

        for run in range(0,m):
            # initialize a random orthogonal matrix
            randVt = ortho_group.rvs(features_pca.shape[1])

            # divide the features dimensions equally among the subspaces
            dpProj = randVt[:,range_value[0]:range_value[1]]

            # projection to the subspace
            X = np.matmul(features_pca,dpProj)

            # rerun the kmeans on the initial clusters on the projected data
            cluster_centers_subspace = cluster_centers
            kmeans_X = KMeans(init=cluster_centers_subspace,n_clusters=nrOfclusters).fit(X)
            cluster_labels = kmeans_X.labels_

            # map the data points onto the clusters
            clusters = {}
            for cluster in range(0, nrOfclusters):
                clusters[cluster] = []

            for i in range(0,len(cluster_labels)):
                 clusters[cluster_labels[i]].append(X[i])

            # update the means of the clusters based on the data points
            updated_cluster_centers = {}
            for c in range(0,len(clusters)):
                updated_cluster_centers[c] = [float(sum(col))/len(col) for col in zip(*clusters[c])]

            # converting updated cluster centers into np array
            cluster_center_values = updated_cluster_centers.values()
            updated_centers = np.array(cluster_center_values)

            # set the old means to new means
            cluster_centers = updated_centers

            # rerun k means on the data points using updated means
            kmeans_XC = KMeans(init=updated_centers, n_clusters=nrOfclusters).fit(X)
            updated_cluster_labels = kmeans_XC.labels_

        # map the data points to the converged clusters
        predicted_clusters_indices = {}
        for cluster in range(0,nrOfclusters):
            predicted_clusters_indices[cluster] = []

        for l in range(0,len(updated_cluster_labels)):
            predicted_clusters_indices[updated_cluster_labels[l]].append(l)
        predicted_clusters[subspace] = predicted_clusters_indices

    # for Pair Counting F-1 Measure calculation
    true_labels_list = []
    for index in true_clusters_indices.keys():
        true_labels_list.append(list(itertools.combinations(true_clusters_indices[index],2)))

    pred_labels_list = []
    pred_indices = []
    for sub in predicted_clusters.keys():
        for index in predicted_clusters[sub].keys():
            pred_indices.append(predicted_clusters[sub][index])
            pred_labels_list.append(list(itertools.combinations(predicted_clusters[sub][index], 2)))

    data_indices = []
    for i in range(0,900):
        data_indices.append(i)

    data_labels = list(itertools.combinations(data_indices,2))

    true_labels = []
    for label in true_labels_list:
        for i in range(0,len(label)):
            true_labels.append(label[i])

    pred_labels = []
    for label in pred_labels_list:
        for i in range(0, len(label)):
            pred_labels.append(label[i])

    true_set = set()
    for label in true_labels:
        true_set.add(label)

    pred_set = set()
    for label in pred_labels:
        pred_set.add(label)

    data_set = set()
    for label in data_labels:
        data_set.add(label)

    a = pred_set.intersection(true_set)
    b = pred_set.difference(true_set)
    c = true_set.difference(pred_set)

    #print len(a)
    #print len(b)
    #print len(c)
    F1_Measure = float(float(2 * len(a))/float((2 * len(a)) + len(b) + len(c)))

    print "Pair Counting F1-Measure: "
    print F1_Measure

    # for variation of information calculation
    print "Average Variation of Information: "
    print float(variation_of_information(pred_indices,true_clusters_indices.values()))/float(nrOfclusters*nrOfSubspaces)

if __name__ == '__main__':
    # defining the way i want to capture user input
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset',
                      default='',  # default empty!
                      help='location of the dataset')
    parser.add_option('--clusters', dest='clusters',
                      default='',  # default empty!
                      help='number of clusters per subspace')
    parser.add_option('--pca', dest='pca_components',
                      default='',  # default empty!
                      help='number of components for PCA')
    (options, args) = parser.parse_args()

    # assigning the user input
    dataset = options.dataset
    clusters = options.clusters
    pca_components = options.pca_components
    pca_components = int(pca_components)

    clustering(dataset,clusters,pca_components)