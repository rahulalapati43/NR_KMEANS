from scipy.stats import ortho_group
import numpy as np
import optparse
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
import itertools

def assignment_step(val,k,randVt,dpProj,cluster_centers,cluster_means):
    # transpose the random orthogonal matrix
    Vt = randVt.transpose()
    # transpose the projection onto the subspace
    Pt = dpProj.transpose()
    # mutliply
    PtVt = np.matmul(Pt, Vt)

    labels = {}
    clusters = {}
    for i in range(0, k):
        clusters[i] = []

    # assign the data points to the clusters based on the cost function
    for x in range(0,len(dpProj)):
        PtVtx = np.matmul(PtVt, dpProj[x])

        # calculate costs
        costs = {}
        for i in range(0,k):
            mean = cluster_centers[val][i]
            PtVtU = np.matmul(PtVt,mean)
            costs[i] = abs(PtVtx - PtVtU) * abs(PtVtx - PtVtU)

        # assigning the data points to a cluster label based on minimum cost
        labels[x] = min(costs.items(), key=lambda l: l[1])[0]
        clusters[labels[x]] = clusters[labels[x]].append(dpProj[x])

        return labels, clusters

def updateclusterStatistics(val,old_cluster_means,assigned_clusters):
    # update cluster means based on the new clusters
    updated_cluster_means = {}
    for i in range(0,len(assigned_clusters)):
        temp_list = assigned_clusters[i]
        sum = 0.0
        len = 0
        for data in temp_list:
            sum = sum + sum(data)
            len = len + len(data)

        updated_cluster_means[i] = sum/float(len)

    return updated_cluster_means

def has_converged(old_cluster_means, new_cluster_means):
    return (set([tuple(a) for a in new_cluster_means]) == set([tuple(a) for a in old_cluster_means]))

def nrkmeans_algo(dataset,clusters):
    # open the input dataset and read the contents of it
    df = open(dataset,'r')
    df_lines = df.readlines()

    clusters_list = clusters.split(',')
    clusters_list = map(int,clusters_list)

    # get the number clusters per subspace
    ks = tuple(clusters_list)
    nrOfSubspaces = len(ks)

    # get the features and labels (cluster centers lables)
    features = {}
    labels = {}

    for i in range(0,len(df_lines)):
        row = df_lines[i].strip()
        row_list = row.split(';')
        labels[i] = row_list[0:3]
        features[i] = map(int,row_list[3:len(row_list)])

    nrOfDims = len(features[0])

    #initialize a random orthogonal matrix
    randVt = ortho_group.rvs(nrOfDims)
    np.dot(randVt, randVt.T) #to check for orthogonality of randVt

    features_list = features.values()
    features_array = np.array([np.array(xi) for xi in features_list])
    rotData = np.matmul(features_array,randVt)

    # mj calculation
    initMSize = nrOfDims/nrOfSubspaces

    # generate mapping Pj to the features and initialize means for all the clusters
    firstDim = 0
    ranges = {}
    means = {}
    for k in range(0,nrOfSubspaces):
        if (k == nrOfSubspaces-1):
            endOfRange = nrOfDims
        else:
            endOfRange = firstDim + initMSize

        ranges[k] = [firstDim, endOfRange]
        firstDim = firstDim + initMSize

        mean_values = []
        for i in range(0,ks[k]):
            mean_values.append(features_array[np.random.randint(features_array.shape[0],size=1)][0])
        means[k] = mean_values

    cluster_means = {}
    for key in means.keys():
        cluster_means[key] = np.array([xi for xi in means[key]])

    #optimization
    old_cluster_means = {}
    cluster_centers = {}
    # per subspace
    for val in range(0,len(ks)):
        range_value = ranges[val]
        k = ks[val]
        # divide the features dimensions equally among the subspaces
        dpProj = rotData[:,range_value[0]:range_value[1]]

        # initialize cluster centers using k-means++
        kmeans = KMeans(n_clusters=k,init="k-means++").fit(dpProj)
        cluster_centers[val] = kmeans.cluster_centers_
        old_cluster_means[val] = cluster_centers[val]

        # run the assignment and update steps until convergence
        while not has_converged(old_cluster_means[val],cluster_means[val]):
            old_cluster_means[val] = cluster_means[val]
            # assigning datapoints to clusters
            assigned_labels, assigned_clusters = assignment_step(val,k,randVt,dpProj,cluster_centers,cluster_means)
            # updating cluster means
            cluster_means[val] = updateclusterStatistics(val,old_cluster_means,assigned_clusters)

    #updating V random orthogonal matrix by creating combined subspaces
    indexes = []
    for i in range(0,len(ks)):
        indexes.append(i)

    #generate possible combinations of subspaces
    combs = list(itertools.combinations(indexes,2))
    for comb in combs:
        range_value = ranges[comb[0]]
        Ps = rotData[:,range_value[0]:range_value[1]]
        range_value = ranges[comb[1]]
        Pt = rotData[:, range_value[0]:range_value[1]]

        #creating a combined projection
        PsPt_list = []
        for i in range(0,len(Ps)):
             PsPt_list.append(np.concatenate((Ps[i],Pt[i]),axis=None))

        PsPt = np.array(PsPt_list)

        # transpose the combined projection
        PsPtT = PsPt.transpose()

        # transpose the random orthogonal matrix
        Vt = randVt.transpose()

        res = np.matmul(PsPtT,Vt)
        res = np.matmul(res,randVt)
        res = np.matmul(res,PsPt)

        # get eigen and real values
        eigenvalues = np.linalg.eigvals(res)
        realvalues = np.real(res)

        for eigvalue in eigenvalues:
            for i in range(0,len(eigvalue)):
                if (eigvalue[i] < 0):
                    eigvalue[i] = 0

        randVt = np.matmul(randVt,realvalues)

    return assigned_clusters,randVt,initMSize,labels,assigned_labels

if __name__ == '__main__':
    # defining the way i want to capture user input
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset',
                      default='',  # default empty!
                      help='location of the dataset')
    parser.add_option('--clusters', dest='clusters',
                      default='',  # default empty!
                      help='number of clusters per subspace')
    (options, args) = parser.parse_args()

    # assigning the user input
    dataset = options.dataset
    clusters = options.clusters

    clusters,rotmatrix,dimensionalities,true_labels,pred_labels = nrkmeans_algo(dataset,clusters)

    print "Resulting Clusters are: \n"
    print clusters

    print "Rotation Matrix is: \n"
    print rotmatrix

    print "Resulting Dimensionalities are: \n"
    print dimensionalities

    # evaluating Pair Counting F1 Measure
    F1_result = precision_recall_fscore_support(true_labels,pred_labels)

    precision = F1_result[0]
    recall = F1_result[1]
    f1_score = F1_result[2]

    print "Pairwise Counting F1-Score: \n"
    print f1_score
