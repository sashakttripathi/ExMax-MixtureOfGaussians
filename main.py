import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm

style.use('fivethirtyeight')
np.random.seed(0)

x = np.linspace(-4, 4, num=20)
x0 = x * np.random.rand(len(x)) + 12    # makes 20 points with mean around x = 12
x1 = x * np.random.rand(len(x)) - 12    # makes 20 points with mean around x = -12
x2 = x * np.random.rand(len(x))         # makes 20 points with mean around x = 0

x_sample = np.stack((x0, x1, x2)).flatten()


class GaussianMixture:
    def __init__(self, x_, iterations):
        self.iterations = iterations
        self.x_ = x_
        self.mue = None
        self.pi = None
        self.var = None

    def find_clusters(self):
        # initializing mue, pi, var
        # self.mue = [-4, 5, 2]         # initialization 1
        self.mue = [-8, 5, 2]           # initialization 2
        self.pi = [1/3, 1/3, 1/3]       # assuming equally distributed clusters
        self.var = [4, 2, 1]

        for i in range(self.iterations):
            # E step
            r = np.zeros((len(x_sample), 3))

            for j, n, p in zip(range(3), [norm(loc=self.mue[0], scale=self.var[0]),
                                          norm(loc=self.mue[1], scale=self.var[1]),
                                          norm(loc=self.mue[2], scale=self.var[2])], self.pi):
                r[:, j] = p * n.pdf(x_sample)       # assign probability with which x belongs to each gaussian

            # normalize the probabilities
            for j in range(len(r)):
                r[i] = r[i] / (np.sum(self.pi)*np.sum(r, axis=1)[i])

            fig = plt.figure(figsize=(10, 10))
            fig_ix0 = fig.add_subplot(111)

            for j in range(len(r)):
                fig_ix0.scatter(self.x_[j], 0, c=np.array([r[j][0], r[i][1], r[i][2]]), s=100)

            # plotting the guessed gaussians

            for n, c in zip([norm(loc=self.mue[0], scale=self.var[0]).pdf(np.linspace(-20, 20, num=60)),
                            norm(loc=self.mue[1], scale=self.var[1]).pdf(np.linspace(-20, 20, num=60)),
                            norm(loc=self.mue[2], scale=self.var[2]).pdf(np.linspace(-20, 20, num=60))],
                            ['r', 'g', 'b']):
                fig_ix0.plot(np.linspace(-20, 20, num=60), n, c=c)

            # M step

            w_c = []

            for j in range(len(r[0])):
                w = np.sum(r[:, j])
                w_c.append(w)

            # calculate the new fraction of total samples in each distribution

            for j in range(len(w_c)):
                self.pi[j] = w_c[j]/np.sum(w_c)

            self.mue = np.sum(self.x_.reshape(len(self.x_), 1)*r, axis=0) / w_c

            var_c = []

            for j in range(len(r[0])):
                var_c.append((1/w_c[j]) * np.dot(((np.array(r[:, j]).reshape(60, 1)) *
                                                  (self.x_.reshape(len(self.x_), 1) - self.mue[j])).T,
                                                 (self.x_.reshape(len(self.x_), 1) - self.mue[j])))

            plt.show()


Gm = GaussianMixture(x_sample, 10)
Gm.find_clusters()
