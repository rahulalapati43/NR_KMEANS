Implementation of Discovering Non-Redundant K-means Clusterings in Optimal Subspaces

Group GG6:
1. Rahul Alapati
2. Aditya Mahajan
3. Sathish Akula

This package consists of the Source and Datasets directories, an Output file, a Project Demo Video and a Project Report. 

The Source directory consists of our python implementations of the NR-KMEANS algorithm, namely Subspace_Clustering.py and our PCA version of the NR-KMEANS, namely Subspace_Clustering_PCA.py.

The Datasets directory consists of ALOI-2Sub, ALOI-2Sub_PCA, Stickfigures, Shuttle, Spam  and Syn3Sub datasets. The experiments were performed on the ALOI-2Sub and Stickfigures dataset. The runtime has been tracked manually, by clocking time.

The Output file consists of the evaluations measures and the runtime of our algorithm on different datasets.


Configuration for experiments: We have run the experiments on a machine with 16 GB RAM and Intel I7 7th Generation processor.


Instructions to Run the Subspace_Clustering.py and Subspace_Clustering_PCA.py:

Please make sure you have python 2.7, numpy, scipy and sklearn installed on your machine. 

1. To view help:

python Subspace_Clustering.py --help
Usage: Subspace_Clustering.py [options]

Options:
  -h, --help           show this help message and exit
  --dataset=DATASET    location of the dataset
  --clusters=CLUSTERS  number of clusters per subspace

python Subspace_Clustering_PCA.py --help
Usage: Subspace_Clustering_PCA.py [options]

Options:
  -h, --help            show this help message and exit
  --dataset=DATASET     location of the dataset
  --clusters=CLUSTERS   number of clusters per subspace
  --pca=PCA_COMPONENTS  number of components for PCA

2. To run using ALOI-2Sub with full dimensions:

python Subspace_Clustering.py --dataset=dataset/aloiIntro.data --clusters=2,2

3. To run using Stickfigures with full dimensions (Scalability test on this dataset with 900 datapoints):

python Subspace_Clustering.py --dataset=dataset/stickfigures_3sub.data --clusters=3,3

4. To run using ALOI-2Sub with reduced dimensions (applying pca to reduce to 8 dimensions):

python Subspace_Clustering_PCA.py --dataset=dataset/aloiIntro.data --clusters=2,2 --pca=8

5. To run using Stickfigures with reduced dimensions (applying pca to reduce to 5 dimensions):

python Subspace_Clustering_PCA.py --dataset=dataset/stickfigures_3sub.data --clusters=3,3 --pca=5

The ouput of all the above codes will be the evaluation measures namely Pair Counting F1-Measure and the Average Variance of Information.