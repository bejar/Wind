"""
.. module:: Results

Results
*************

:Description: Results

    

:Authors: bejar
    

:Version: 

:Created on: 07/05/2018 13:50 

"""

__author__ = 'bejar'

from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp
import seaborn as sn

plt.style.use('ggplot')
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def find_exp(query):
    return col.find(query)


def count_exp(query):
    print(col.count(query))


def compare_exp(lexp, n=10):
    fig = plt.figure(figsize=(16, 10), dpi=100)

    axes = fig.add_subplot(1, 2, 1)

    ldiff = {'regdir': [], 'seq2seq': [], 'mlps2s': [], 'convo': [], 'mlpdir': [], 'svmdir': []}

    for exp in lexp:
        data = np.array(exp['result'])
        ldiff[exp['arch']['mode']].append((np.sum(data[:, 1]), exp))
        axes.plot(data[:, 0], data[:, 1], colors[exp['arch']['mode']])

    axes = fig.add_subplot(1, 2, 2)
    besta = {}
    for type in ldiff:
        besta[type] = [b[1] for b in sorted(ldiff[type], reverse=True)[:n]]

        for exp in besta[type]:
            data = np.array(exp['result'])
            axes.plot(data[:, 0], data[:, 1], colors[exp['arch']['mode']])
            axes.plot(data[:, 0], data[:, 3], colors[exp['arch']['mode']] + '--')

    plt.show()
    return besta


def graph_compare_exp(lexp, n=10, lag=0):
    fig = plt.figure(figsize=(6, 4), dpi=100)

    ldiff = {'regdir': [], 'seq2seq': [], 'mlps2s': [], 'convo': [], 'mlpdir': [], 'svmdir': []}

    for exp in lexp:
        data = np.array(exp['result'])
        ldiff[exp['arch']['mode']].append((np.sum(data[:, 1]), exp))

    axes = fig.add_subplot(1, 1, 1)
    besta = {}
    for type in ['regdir', 'seq2seq', 'mlpdir', 'mlps2s', 'svmdir']:
        besta[type] = [b[1] for b in sorted(ldiff[type], reverse=True)[:n]]

        zax = np.zeros((n, 12))
        z1 = np.zeros((n, 12))
        z3 = np.zeros((n, 12))
        for i, exp in enumerate(besta[type]):
            data = np.array(exp['result'])
            # print(data)
            zax[i] = data[:, 0].T
            z1[i] += data[:, 1].T
            z3[i] += data[:, 3].T

        zmean1 = np.mean(z1, axis=0)
        zmean3 = np.mean(z3, axis=0)

        axes.plot(zax[0, lag:], zmean1[lag:], colors[type], label=labels[type])
        axes.plot(zax[0, lag:], zmean3[lag:], colors[type] + '--', label=labels[type])

    plt.legend()
    plt.xlabel('hours')
    plt.ylabel(r'$R^2$')
    plt.show()

    fig.savefig('experiments.pdf', orientation='landscape', format='pdf')
    return besta


def test_compare_exp(lexp, comp, n=10):
    ldiff = {'regdir': [], 'seq2seq': [], 'mlps2s': [], 'convo': [], 'mlpdir': [], 'svmdir': []}

    for exp in lexp:
        data = np.array(exp['result'])
        ldiff[exp['arch']['mode']].append((np.sum(data[:, 1]), exp))

    besta = {}
    for type in comp:
        besta[type] = [b[1] for b in sorted(ldiff[type], reverse=True)[:n]]

    for i in range(12):
        l1 = []
        l2 = []
        for v in besta[comp[0]]:
            data = np.array(v['result'])
            l1.append(np.array(data[i, 1]))
        for v in besta[comp[1]]:
            data = np.array(v['result'])
            l2.append(np.array(data[i, 1]))

        print(i, ks_2samp(l1, l2).pvalue)
        print(np.mean(l1), np.mean(l2), np.mean(l1) - np.mean(l2))


def best_parameters(lexp, archtype, n):
    ldiff = {'regdir': [], 'seq2seq': [], 'mlps2s': [], 'convo': [], 'mlpdir': [], 'svmdir': []}

    for exp in lexp:
        data = np.array(exp['result'])
        ldiff[exp['arch']['mode']].append((np.sum(data[:, 1]), exp))

    best = {}
    for atype in ['regdir', 'seq2seq', 'mlpdir', 'mlps2s', 'svmdir']:
        best[atype] = [b[1] for b in sorted(ldiff[atype], reverse=True)[:n]]

    for i in range(n):
        print('ID:', best[archtype][i]['_id'])
        print('DS:', best[archtype][i]['data']['dataset'])
        print('LAG:', best[archtype][i]['data']['lag'])
        print('VARS:', best[archtype][i]['data']['vars'])
        if 'svm' in best[archtype][i]['arch']['mode']:
            print('MODE:', best[archtype][i]['arch']['mode'])
            print('C:', best[archtype][i]['arch']['C'])
            print('Kernel:', best[archtype][i]['arch']['kernel'])
            print('Eps:', best[archtype][i]['arch']['epsilon'])
            print('Deg:', best[archtype][i]['arch']['degree'])
            print('coef0:', best[archtype][i]['arch']['coef0'])
        else:
            print('MODE:', best[archtype][i]['arch']['mode'])
            print('RNN:', best[archtype][i]['arch']['rnn'] if 'rnn' in best[archtype][i]['arch'] else '')
            print('NLY:', best[archtype][i]['arch']['nlayers'])
            print('NLYE:', best[archtype][i]['arch']['nlayersE'] if 'nlayersE' in best[archtype][i]['arch'] else '')
            print('NLYD', best[archtype][i]['arch']['nlayersD'] if 'nlayersD' in best[archtype][i]['arch'] else '')
            print('NNEUR:', best[archtype][i]['arch']['neurons'] if 'neurons' in best[archtype][i]['arch'] else '')
            print(
            'NNEURD:', 0 if not 'neuronsD' in best[archtype][i]['arch'] else best[archtype][i]['arch']['neuronsD'])
            print('DROP:', best[archtype][i]['arch']['drop'])
            print(
            'BI:', best[archtype][i]['arch']['bidirectional'] if 'bidirectional' in best[archtype][i]['arch'] else '')
            print('FILT:', best[archtype][i]['arch']['filters'] if 'filters' in best[archtype][i]['arch'] else '')
            print('STRD:', best[archtype][i]['arch']['strides'] if 'strides' in best[archtype][i]['arch'] else '')
            print(
            'KERSIZ:', best[archtype][i]['arch']['kernel_size'] if 'kernel_size' in best[archtype][i]['arch'] else '')
            print('ACT:', best[archtype][i]['arch']['activation'])
            print('ACTR:', best[archtype][i]['arch']['activation_r'])
            print('OPT:', best[archtype][i]['training']['optimizer'])
            print('FULLLY:', best[archtype][i]['arch']['full'])

        print(np.array(best[archtype][i]['result'])[:, 1])


def exp_distrib(lexp):
    ldiff = []
    for exp in lexp:
        data = np.array(exp['result'])
        ldiff.append((data[:, 1]))

    ldiff = np.array(ldiff)
    print(ldiff.shape)

    sn.boxplot(data=ldiff)
    plt.show()


if __name__ == '__main__':
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    colors = {'regdir': 'b', 'seq2seq': 'r', 'mlps2s': 'g', 'convo': 'c', 'mlpdir': 'y', 'svmdir': 'm'}
    labels = {'regdir': 'RNNdir', 'seq2seq': 'RNNs2s', 'mlps2s': 'MLPs2s',
              'convo': 'CNNdir', 'mlpdir': 'MLPdir', 'svmdir': 'SVMdir'}
    # query1= {'status':'done',
    #      #'arch.mode':'seq2seq',
    #          #   'data.lag':32,
    #          # 'data.dataset':3,
    #            # 'data.vars': [0,1,2,3],
    #           #  'arch.neurons': 32,
    #            # 'arch.drop':0.1,
    #           #  'arch.rnn':'GRU',
    #            # 'arch.bidirectional':False,
    #            # 'arch.activation':'tanh'
    #            }
    #
    # count_exp(query1)
    # res1 = find_exp(query1)
    # #best = graph_compare_exp(res1, n=20)
    # # test_compare_exp(res1, ['regdir', 'mlps2s'], n=20)
    # best_parameters(res1, 'regdir', 5)
    #
    query2 = {'status': 'done', "arch.mode": "regdir", "experiment.type": "expbest"}

    count_exp(query2)
    res1 = find_exp(query2)

    exp_distrib(res1)

    query2 = {'status': 'done', "arch.mode": "mlps2s", "experiment.type": "expbest"}

    count_exp(query2)
    res1 = find_exp(query2)

    exp_distrib(res1)
