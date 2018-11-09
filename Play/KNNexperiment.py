"""
.. module:: KNNexperiment

KNNexperiment
*************

:Description: KNNexperiment

    

:Authors: bejar
    

:Version: 

:Created on: 29/10/2018 11:50 

"""
from sklearn.neighbors import KDTree
from Wind.Config.Paths import wind_data_path
from Wind.Data.DataSet import Dataset
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'bejar'

if __name__ == '__main__':
    data = {
        "datanames": [
            "95-47500-12"
        ],
        "vars": [0, 1, 2, 3],
        "datasize": 40912,
        "testsize": 20456,
        "dataset": 0,
        "lag": 48,
        "ahead": [1, 12]
    }

    mode = 's2s'
    iahead = 1
    fahead = 12

    dataset = Dataset(config=data, data_path=wind_data_path)

    dataset.generate_dataset(ahead=[iahead, fahead], mode=mode)

    print(dataset.summary())

    train_x = dataset.train_x
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
    train_y = dataset.train_y
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1]))

    print(train_x.shape, train_y.shape)
    tree = KDTree(train_x, leaf_size=1)
    nneigh = 20
    for s in range(train_x.shape[0]):
        # radius = 0.3
        # neigh = tree.query_radius(train_x[s].reshape(1, -1), r=radius, count_only=False, return_distance=True,
        #                               sort_results=True)
        #
        # while neigh[0][0].shape[0] < 2 and radius < 1.2:
        #     radius += 0.1
        #     neigh = tree.query_radius(train_x[s].reshape(1, -1), r=radius, count_only=False, return_distance=True,
        #                               sort_results=True)
        neigh = tree.query(train_x[s].reshape(1, -1), k=nneigh, return_distance=True,
                                  sort_results=True)
        print(neigh)

        if neigh[1][0].shape[0] > 1:
            # print(radius, neigh[0][0].shape[0])

            fig = plt.figure()

            axes = fig.add_subplot(1, 2, 1)
            plt.plot(train_x[s][::1], 'k--')
            c=0
            for i in neigh[1][0][1:]:
                if i<s-data['lag'] or i>s+data['lag']:
                    plt.plot(train_x[i][::1])
                    c+=1
                if c==3:
                    break
            plt.ylim([-2, 2])

            axes = fig.add_subplot(1, 2, 2)
            plt.plot(train_y[s], 'k--')
            c=0
            for i in neigh[1][0][1:]:
                if i<s-data['lag'] or i>s+data['lag']:
                    plt.plot(train_y[i])
                    c+=1
                if c==3:
                    break

            plt.ylim([-2, 2])
            plt.show()
