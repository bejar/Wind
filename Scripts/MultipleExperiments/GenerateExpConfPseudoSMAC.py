"""
.. module:: GenerateExpConfPseudoSMAC

GenerateExpConfPseudoSMAC
*************

:Description: GenerateExpConfPseudoSMAC

  Attempt to implement similar SMAC algorithm for meta parameter exploration
    

:Authors: bejar
    

:Version: 

:Created on: 08/04/2019 14:19 

"""

import argparse
from time import time

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
from copy import deepcopy
from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

__author__ = 'bejar'


def change_config(config, param, sites):
    """
    Substitutes default parameter values by generated parameters
    """
    lconf = []
    for s in sites:
        conf = deepcopy(config)
        for sec in param:
            for p in param[sec]:
                conf[sec][p] = param[sec][p]
        conf['data']['datanames'] = [f'{s}-12']
        conf['site'] = s
        lconf.append(conf)
    return lconf


def hash_config(conf):
    """
    Creates a canonical string representation for a configuration
    by traversing alphabetically the dictionary and appending a representation string
    :param conf:
    :return:
    """
    s = ""
    for sec in sorted(conf.keys()):
        if list(conf[sec].keys()):
            for p in sorted(conf[sec].keys()):
                s += f"{p}={conf[sec][p]}/"
            s += '#'
    return s

def hash_experiment(exp, confP):
    """
    Creates a canonical string representation for a experiment configuration given the configuration template
    by traversing alphabetically the dictionary and appending a representation string
    :param conf:
    :return:
    """
    s = ""
    for sec in sorted(confP.keys()):
        if list(confP[sec].keys()):
            for p in sorted(confP[sec].keys()):
                s += f"{p}={exp[sec][p]}/"
            s += '#'
    return s



def generate_random_config(config):
    """
    generates a random config by picking parameters from the configuration values
    """
    conf = {}
    for sec in config:
        conf[sec] = {}
        for p in config[sec]:
            conf[sec][p] = config[sec][p][np.random.choice(len(config[sec][p]))]
    return conf


def recode_dataframe(df, conf):
    """
    Recodes the values of the dataframe so it can be use with scikit learn
    :param df:
    :param config:
    :return:
    """
    for sec in sorted(conf.keys()):
        for p in sorted(conf[sec].keys()):
            df[p] = df[p].replace(to_replace=[str(v) for v in conf[sec][p]],
                                  value=[i for i in range(len(conf[sec][p]))])
    return df


def get_df_configurations(df, conf, as_set=True):
    """
    Generates a set with the configurations already tested in canonical form (set or list)

    :param df:
    :param config:
    :return:
    """
    if as_set:
        sconf = set()
    else:
        sconf = []

    for i in range(len(df)):
        s = ""
        for sec in sorted(conf.keys()):
            for p in sorted(conf[sec].keys()):
                s += f"{p}={df.iloc[i][p][0]}/"
            s += '#'
        if as_set:
            sconf.add(s)
        else:
            sconf.append(s)
    return sconf


def config_to_example(conf, confP, vars):
    """
    Transforms a configuration into a recoded example
    :param conf:
    :return:
    """
    a = np.zeros(len(vars))
    for sec in sorted(conf.keys()):
        for p in sorted(conf[sec].keys()):
            a[vars.index(p)] = confP[sec][p].index(conf[sec][p])

    return a.reshape((1, -1))


def retrieve_results_as_dataframe(experiment):
    """
    Retrieves all the results for an experiment as a dataframe

    :return:
    """
    lvars = ['hour', 'site', 'test', 'val'] + arch + data + train
    ddata = {}
    for var in lvars:
        ddata[var] = []

    # Retrieve all results
    lexp = col.find({'experiment': experiment})
    for exp in lexp:
        # To maintain backwards compatibility
        if 'result' in exp:
            exdata = np.array(exp['result'])
        elif 'results' in exp:
            exdata = np.array(exp['results'])

        for i in range(exdata.shape[0]):
            lvals = [i + 1]
            lvals.append(int(exp['data']['datanames'][0].split('-')[1]))
            lvals.append(exdata[i, 1])
            lvals.append(exdata[i, 2])

            for v in arch:
                lvals.append(str(exp['arch'][v]))

            for v in data:
                lvals.append(str(exp['data'][v]))

            for v in train:
                lvals.append(str(exp['training'][v]))

            for var, val in zip(lvars, lvals):
                ddata[var].append(val)

    # Transform the experiment results into a dataframe and get the experiment mean R^2
    exp_df = pd.DataFrame(ddata)
    exp_df = exp_df.groupby(by=['site'] + arch + data + train, as_index=False).sum()
    exp_df.drop(columns=['hour', 'site'], inplace=True)
    exp_df = exp_df.groupby(by=arch + data + train, as_index=False).agg({'test': ['mean', 'count'], 'val': ['mean']})

    return exp_df


def regenerate_conf(df, i):
    """
    Converts a configuration from a dataframe position to a configuration dictionary
    :param df:
    :param i:
    :return:
    """

    conf = {'arch': {}, 'train': {}, 'data': {}}
    for var, sec in zip([arch, data, train],['arch', 'data', 'train']):
        for v in var:
            try:
                conf[sec][v] = eval(df.iloc[i][v][0])
            except:
                conf[sec][v] = df.iloc[i][v][0]

    return conf


def concat_sites(sites, i, f):
    """
    Concatenates batches of sites
    :param sites:
    :param i:
    :param f:
    :return:
    """
    lsites = []
    for v in range(i,f):
        print(v)
        lsites.extend(sites[v])
    return lsites


def insert_configurations(lconf, sites, tstamp=None):
    """
    Inserts a batch of configurations for a batch of sites

    :param lconf:
    :param sites:
    :return:
    """
    lsitesconf = []
    for c in lconf:
        lsitesconf += change_config(configB, c, sites)

    ids = int(time()*10000)
    if tstamp is None:
        tstamp = str(ids)

    print(f'TSTAMP={tstamp}')

    print(f'Inserting {len(lsitesconf)} configurations on the database')
    for n, sc in tqdm(enumerate(lsitesconf)):
        sc['timestamp'] = tstamp
        sc['experiment'] = args.exp
        sc['status'] = 'pending'
        sc['result'] = []
        site = sc['data']['datanames'][0].split('-')
        sc['_id'] = f"{ids}{n:05d}{int(site[1]):06d}"
        sc['exploration'] = 'SMAC'
        if not args.test:
            col.insert_one(sc)


def get_best_configurations(exp_df, n):
    """
    Return the n best cofigurations
    :param exp_df:
    :return:
    """
    lexp = []
    for i in range(len(exp_df)):
        lexp.append((exp_df.iloc[i]['test']['mean'], i, exp_df.iloc[i]['test']['count']))

    lexp = sorted(lexp, reverse=True)[:n]

    # For the best configurations, regenerate configuration from dataframe
    lbestconf = []
    for _, i, _ in lexp:
        lbestconf.append(regenerate_conf(exp_df, i))
    return lbestconf


def change_one_conf(config, configP):
    """
    Changes randomly one item in the configuration
    Serves as a mutation operator also
    :param conf:
    :return:
    """
    newconf = deepcopy(config)
    att = ''
    while att == '':
        # choose randomly a section
        sec = list(newconf.keys())[np.random.choice(len(list(newconf.keys())))]
        if len(list(newconf[sec].keys())) != 0:
            # choose randomly an attribute
            att = list(newconf[sec].keys())[np.random.choice(len(list(newconf[sec].keys())))]
    # choose randomly a new value
    vals = deepcopy(configP[sec][att])
    vals.remove(newconf[sec][att])
    newconf[sec][att] = vals[np.random.choice(len(vals))]

    return newconf


def crossover_conf(config1, config2, configP, cross, mutate):
    """
    Cross over configurations by interchanging parameters
    For now probability of cross over is 50% (add as a pa

    :param config:
    :param configP:
    :return:
    """
    newconf1 = deepcopy(config1)
    newconf2 = deepcopy(config2)
    for var, sec in zip([arch, data, train],['arch', 'data', 'train']):
        for v in var:
            if (newconf1[sec][v] != newconf2[sec][v]) and (np.random.sample() < cross):
                temp =  newconf1[sec][v]
                newconf1[sec][v] = newconf2[sec][v]
                newconf2[sec][v] = temp
    if np.random.sample() < mutate:
        newconf1 = change_one_conf(newconf1, configP)
    if np.random.sample() < mutate:
        newconf2 = change_one_conf(newconf2, configP)
    return newconf1, newconf2

def build_regression_tree(exp, attributes, configP, nbest):
    """
    Builds the regression tree to predict the configurations performance

    :param exp:
    :return:
    """
    exp_df = retrieve_results_as_dataframe(exp)

    # We have already results to train the regressor
    # Train a random forest regressor to predict accuracy of new configurations
    lbestconf = get_best_configurations(exp_df, nbest)

    exp_df = recode_dataframe(exp_df, configP)
    dataset = exp_df.to_numpy()
    rfr = RandomForestRegressor(n_estimators=1000)
    # -3 is the test prediction
    rfr.fit(dataset[:, :-3], dataset[:, -3])

    pred_max = np.max(dataset[:, -3])
    pred_std = np.std(dataset[:, -3])

    print("Feature importances:")
    limp = zip(rfr.feature_importances_, attributes)
    for i, f in sorted(limp):
        print(f"F({f}) = {i:3.2f}")

    print('------------------------')

    return rfr, pred_max, pred_std, lbestconf

def predict_randomforestregression(rfr, example):
    """
    Computes the prediction of the random forestregression and the stdev of the prediction from the individual predictions
    :param rfr:
    :return:
    """
    pred = rfr.predict(example)
    lpred=[]
    for tree in rfr.estimators_:
        lpred.extend(tree.predict(example))

    return pred, np.std(lpred)



# -------------------------------------
# Strategies for generating configurations


def generate_random_configurations(conf_done, configP, maxtries, maxconf):
    """
    Generates a set of random experiment configurations different from the
    current ones

    Performs maxtries until maxconf configurations are generated
    :return:
    """
    lconf = []
    nc = 0
    for i in tqdm(range(maxtries)):
        conf = generate_random_config(configP)
        if hash_config(conf) not in conf_done:
            if args.print:
                print(conf)
            lconf.append(conf)
            conf_done.add(hash_config(conf))
            nc += 1
        if nc >= maxconf:
            break

    insert_configurations(lconf, concat_sites(smacexp['sites'],0,args.ninitbatches))


def intensify_configurations(expname, maxconf):
    """
    Add a new set ob site batches for the configurations with the best performance

    :param expname:
    :param maxconf:
    :return:
    """
    exp_df = retrieve_results_as_dataframe(expname)

    lexp = []
    for i in range(len(exp_df)):
        lexp.append((exp_df.iloc[i]['test']['mean'], i, exp_df.iloc[i]['test']['count']))

    lexp = sorted(lexp, reverse=True)[:maxconf]
    # For the best configurations, regenerate configuration from dataframe and add experiments
    # for the following batch
    tstamp = str(int(time() * 10000))
    for _, i, count in lexp:
        conf = regenerate_conf(exp_df, i)
        if args.print:
            print(conf)

        # Compute the next batch os sites to be used in the experiments
        # Assumes that most of the sites from the previous run have been executed
        if count % BATCH == 0:
            ibatch = (count // BATCH)
        else:
            ibatch = (count // BATCH) + 1

        insert_configurations([conf], concat_sites(smacexp['sites'], ibatch, ibatch + args.ninitbatches), tstamp=tstamp)


def exploit_random(conf_done, configP, attributes, rfr, pred_max, pred_std, maxtries, maxconf, stdevprop):
    """
    Exploit prediction surface by generating random configurations

    :param conf_done:
    :param configP:
    :param maxtries:
    :param maxconf:
    :return:
    """
    nc = 0
    lconf = []
    print("Scanning configurations ...")
    for i in tqdm(range(maxtries)):
        conf = generate_random_config(configP)
        if hash_config(conf) not in conf_done:
            conf_done.add(hash_config(conf))
            v = config_to_example(conf, configP, attributes)
            # pred = rfr.predict(v)
            pred, p_std = predict_randomforestregression(rfr, v)
            # if pred + (stdevprop * pred_std) > pred_max:
            if pred + (stdevprop * p_std) > pred_max:
                if args.print:
                    print(pred, pred + (stdevprop * p_std))
                    print(conf)
                lconf.append(conf)
                nc += 1
        if nc >= maxconf:
            break
    return lconf


def exploit_local(lbestconf, conf_done, configP, attributes, rfr, pred_max, pred_std, maxtries, maxconf, stdevprop):
    """
    Exploit prediction surface by generating local configurations around the best ones

    :param conf_done:
    :param configP:
    :param attributes:
    :param rfr:
    :param maxtries:
    :param maxconf:
    :return:
    """
    lconf = []
    nc = 0
    for i in tqdm(range(maxtries)):
        newconf = change_one_conf(lbestconf[np.random.choice(len(lbestconf))], configP)
        if hash_config(newconf) not in conf_done:
            conf_done.add(hash_config(newconf))
            v = config_to_example(newconf, configP, attributes)
            # pred = rfr.predict(v)
            pred, p_std = predict_randomforestregression(rfr, v)
            # if pred + (stdevprop * pred_std) > pred_max:
            if pred + (stdevprop * p_std) > pred_max:
                if args.print:
                    print(pred, pred + (stdevprop * p_std))
                    print(newconf)
                lconf.append(newconf)
                # add candidates to list of best configurations to explore more configurations locally
                lbestconf.append(newconf)
                nc += 1
        if nc >= maxconf:
            break
    return lconf


def exploit_genetic(lbestconf, conf_done, configP, attributes, rfr, pred_max, pred_std, maxtries, maxconf, stdevprop, cross, mutate):
    """
    Exploit prediction surface by generating configurations by crossing the best configurations

    :param conf_done:
    :param configP:
    :param attributes:
    :param rfr:
    :param maxtries:
    :param maxconf:
    :param stdevprop:
    :param cross:
    :param mutate:
    :return:
    """
    lconf = []
    nc = 0
    for i in tqdm(range(maxtries)):
        choices = np.random.choice(len(lbestconf), size=2, replace=False)
        newconfs = crossover_conf(lbestconf[choices[0]], lbestconf[choices[1]], configP, cross, mutate)
        for newconf in newconfs:
            if hash_config(newconf) not in conf_done:
                conf_done.add(hash_config(newconf))
                v = config_to_example(newconf, configP, attributes)
                # pred = rfr.predict(v)
                pred, p_std = predict_randomforestregression(rfr, v)
                # if pred + (stdevprop * pred_std) > pred_max:
                if pred + (stdevprop * p_std) > pred_max:
                    if args.print:
                        print(pred, pred + (stdevprop * p_std))
                        print(newconf)
                        print(hash_config(newconf))
                    lconf.append(newconf)
                    # add candidates to list of best configurations to explore more configurations
                    lbestconf.append(newconf)
                    nc += 1
        if nc >= maxconf:
            break
    return lconf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Experiment configuration')
    parser.add_argument('--pconfig', required=True, help='Parameters to explore configuration')

    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')

    parser.add_argument('--exp', required=True, help='Experiment Name')

    parser.add_argument('--npar', type=int, default=10,
                        help='Number of parameter combinations to generate/explore/exploit')
    parser.add_argument('--nbest', type=int, default=10,
                        help='Number of best solutions to use for exploitations')
    parser.add_argument('--std', type=float, default=.4, help='Range for the STD of the best prediction to exploit')
    parser.add_argument('--cross', type=float, default=.5, help='Crossover probability for genetic exploitation')
    parser.add_argument('--mutate', type=float, default=.1, help='Mutation probability for genetic exploitation')


    parser.add_argument('--confexp', type=int, default=2000,
                        help='Number of parameter combinations to test for exploitation')

    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--print', action='store_true', default=False, help='Print configurations')

    # Strategies/Stages
    parser.add_argument('--init', type=int, default=None, help='Initialize the random sites for exploration')
    parser.add_argument('--refexp', default=None, help='Use reference experiment (Name)')

    parser.add_argument('--random', action='store_true', default=False, help='Generate random configurations')
    parser.add_argument('--intensify', action='store_true', default=False, help='Intensify best configurations')
    parser.add_argument('--exploit', default=None, choices=['random', 'local', 'genetic'],
                        help='Use prediction surface for generating new promising configurations')
    parser.add_argument('--ninitbatches', type=int, default=1,
                        help='Number of initial batches of sites for new configurations or intensification')

    args = parser.parse_args()

    # Template configuration file
    configB = load_config_file(args.config, upload=True)
    # Parameters space configuration file (all parameters and values to explore)
    configP = load_config_file(args.pconfig, upload=True)
    # Variables used from the experiments
    arch = list(configP['arch'].keys()) if 'arch' in configP else []
    data = list(configP['data'].keys()) if 'data' in configP else []
    train = list(configP['train'].keys()) if 'train' in configP else []

    # DB stuff
    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    # Check if experiment already exists
    smacexp = col.find_one({'SMAC': 'init', 'smexperiment': args.exp})

    # Initialize experiment
    # 1)  Pick a random sample of sites and divide it in exploration stages
    # Default is 1000 divided in batches of 25
    BATCH = 25
    if args.init is not None:
        if smacexp:
            raise NameError("Experiment already initialized")
        else:
            # Pick random sites and divide it in batches
            lsites = np.random.choice(range(126691), args.init, replace=False)
            lsites = [f'{site // 500}-{site}' for site in lsites]
            lbatches = []
            for i in range(0, len(lsites), BATCH):
                lbatches.append(lsites[i:i + BATCH])
            expconf = {'SMAC': 'init', 'smexperiment': args.exp, 'batch': BATCH, 'sites': lbatches}
            col.insert_one(expconf)

    # Initialize experiment
    # 2) Copy reference experiment configuration
    elif args.refexp is not None:
        if smacexp:
            raise NameError("Experiment already initialized")
        else:
            smacrefexp = col.find_one({'SMAC': 'init', 'smexperiment': args.refexp})
            if not smacrefexp:
                raise NameError("Reference experiment does not exists")
            else:
                expconf = {'SMAC': 'init', 'smexperiment': args.exp,
                           'batch': smacrefexp['batch'], 'sites': smacrefexp['sites']}
                col.insert_one(expconf)
                print(f"Initializing experiment {args.exp}")
    # Experiment already initialized
    else:
        if not smacexp:
            raise NameError("Experiment not initialized")

        # get first site to get all unique configurations (all run configurations have been generated for this site)
        refsite = smacexp['sites'][0][0]
        expsite = col.find_one({'experiment': args.exp, 'site': refsite})

        conf_done = set()
        if expsite is not None:  # Some experiments have been added already
            # Find all the current experiments
            lexp = col.find({'experiment': args.exp, 'site': refsite})
            conf_done = set()
            for exp in lexp:
                conf_done.add(hash_experiment(exp, configP))

        # 1) Generate random configurations
        if args.random or expsite is None:
            generate_random_config(conf_done, configP, args.confexp, args.npar)

        # 2) Generate more experiments for the configurations with higher score
        elif args.intensify:
            intensify_configurations(args.exp, args.npar)

        # 3) Generate more experiments from the prediction of the score
        elif args.exploit:
            rfr, pred_max, pred_std, lbestconf = build_regression_tree(args.exp, arch + data + train, configP, args.nbest)

            # 3.a) randomly by exploring the prediction surface generating and testing random configurations
            if args.exploit == 'random':
                lconf = exploit_random(conf_done, configP, arch + data + train, rfr, pred_max, pred_std, args.confexp, args.npar, args.std)

            # 3.b) take the nbest configurations and generate new configurations by changing randomly
            # one attribute and testing accuracy
            elif args.exploit == 'local':
                lconf = exploit_local(lbestconf, conf_done, configP, arch + data + train, rfr, pred_max, pred_std, args.confexp, args.npar, args.std)

            # 3.c) take the nbest configurations and generate candidates by cross overing
            elif args.exploit == 'genetic':
                lconf = exploit_genetic(conf_done, configP, arch + data + train, rfr, pred_max, pred_std, args.confexp, args.npar, args.std, args.cross, args.mutate)

            # insert promising configurations with a number batches of sites
            if len(lconf) > 0:
                insert_configurations(lconf, concat_sites(smacexp['sites'],0,args.ninitbatches))
