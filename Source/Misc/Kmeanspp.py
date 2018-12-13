import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class KMeans:
    def __init__(self, K, X=None, N=0):
        self.K = K
        if X == None:
            if N == 0:
                raise Exception("If no data is provided, \
                                 a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X = self._init_cluster(N, K)
        else:
            self.X = X
            self.N = len(X)
        self.cen = None
        self.clusters = None
        self.method_used = None

    def _init_cluster(self, N, k):
        n = float(N) / k
        X = []
        for i in range(k):
            c = (random.uniform(-1, 1), random.uniform(-1, 1))
            s = random.uniform(0.05, 0.15)
            x = []
            while len(x) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                # randomly drawing points in the range [-1,1]
                if abs(a) and abs(b) < 1:
                    x.append([a, b])
            X.extend(x)
        X = np.array(X)[:N]
        return X

    def plot_graph(self):
        X = self.X
        plot_fig = plt.figure(figsize=(5, 5))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        if self.cen and self.clusters:
            cen = self.cen
            clus = self.clusters
            K = self.K
            for m, clu in clus.items():
                cs = cm.Spectral(1. * m / self.K)
                plt.plot(cen[m][0], cen[m][1], 'o', marker='*', \
                         markersize=12, color=cs)
                plt.plot(list(zip(*clus[m]))[0], list(zip(*clus[m]))[1], '.', \
                         markersize=8, color=cs, alpha=0.5)
        else:
            plt.plot(list(zip(*X))[0], list(zip(*X))[1], '.', alpha=0.5)

        plot_title = 'K-means Clustering'
        plot_pars = 'N=%s, K=%s' % (str(self.N), str(self.K))
        plt.title('\n'.join([plot_pars, plot_title]), fontsize=16)
        plt.savefig('kpp_N%s_K%s.png' % (str(self.N), str(self.K)), \
                    bbox_inches='tight', dpi=200)

    def _assign_cluster(self):
        cen = self.cen
        clusters = {}
        for x in self.X:
            true_key = min([(i[0], np.linalg.norm(x - cen[i[0]])) \
                             for i in enumerate(cen)], key=lambda t: t[1])[0]
            try:
                clusters[true_key].append(x)
            except KeyError:
                clusters[true_key] = [x]
        self.clusters = clusters

    def _calculate_new_centers(self):
        clusters = self.clusters
        newcen = []
        keys = sorted(self.clusters.keys())
        for k in keys:
            newcen.append(np.mean(clusters[k], axis=0))
        self.cen = newcen

    def _has_converged(self):
        K = len(self.oldcen)
        return (set([tuple(a) for a in self.cen]) == \
                set([tuple(a) for a in self.oldcen]) \
                and len(set([tuple(a) for a in self.cen])) == K)

    def cluster_centers(self, method_used='random'):
        self.method_used = method_used
        X = self.X
        K = self.K
        self.oldcen = random.sample(X.tolist(), K)
        if method_used != '++':
            # Initialize to K random centers
            self.cen = random.sample(X.tolist(), K)
        while not self._has_converged():
            self.oldcen = self.cen
            # Assign all points in X to clusters
            self._assign_cluster()
            # Reevaluate centers
            self._calculate_new_centers()


def main():
    kmeans = KMeans(K=3, N=200)
    kmeans.cluster_centers()
    kmeans.plot_graph()


if __name__ == "__main__":
    main()

