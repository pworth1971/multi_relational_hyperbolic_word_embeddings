# -*- coding: utf-8 -*-

"""
 Functions for fetching similarity datasets
"""

import os

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from .utils import _get_as_pd, _fetch_file


def fetch_MTurk():
    """
    Fetch MTurk dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    References
    ----------
    Radinsky, Kira et al., "A Word at a Time: Computing Word Relatedness Using Temporal Semantic Analysis", 2011

    Notes
    -----
    Human labeled examples of word semantic relatedness. The data pairs were generated using an algorithm as
    described in the paper by [K. Radinsky, E. Agichtein, E. Gabrilovich, S. Markovitch.].
    Each pair of words was evaluated by 10 people on a scale of 1-5.

    Additionally scores were multiplied by factor of 2.
    """
    data = _get_as_pd('https://www.dropbox.com/s/f1v4ve495mmd9pw/EN-TRUK.txt?dl=1',
                      'similarity', header=None, sep=" ").values
    return Bunch(X=data[:, 0:2].astype("object"),
                 y=2 * data[:, 2].astype(float))


def fetch_MEN(which="all", form="natural"):
    """
    Fetch MEN dataset for testing similarity and relatedness

    Parameters
    ----------
    which : "all", "test" or "dev"
    form : "lem" or "natural"

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores

    References
    ----------
    Published at http://clic.cimec.unitn.it/~elia.bruni/MEN.html.

    Notes
    -----
    Scores for MEN are calculated differently than in WS353 or SimLex999.
    Furthermore scores where rescaled to 0 - 10 scale to match standard scaling.

    The MEN Test Collection contains two sets of English word pairs (one for training and one for testing)
    together with human-assigned similarity judgments, obtained by crowdsourcing using Amazon Mechanical
    Turk via the CrowdFlower interface. The collection can be used to train and/or test computer algorithms
    implementing semantic similarity and relatedness measures.
    """
    if which == "dev":
        data = _get_as_pd('https://www.dropbox.com/s/c0hm5dd95xapenf/EN-MEN-LEM-DEV.txt?dl=1',
                          'similarity', header=None, sep=" ")
    elif which == "test":
        data = _get_as_pd('https://www.dropbox.com/s/vdmqgvn65smm2ah/EN-MEN-LEM-TEST.txt?dl=1',
                          'similarity/EN-MEN-LEM-TEST', header=None, sep=" ")
    elif which == "all":
        data = _get_as_pd('https://www.dropbox.com/s/b9rv8s7l32ni274/EN-MEN-LEM.txt?dl=1',
                          'similarity', header=None, sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    if form == "natural":
        # Remove last two chars from first two columns
        data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    elif form != "lem":
        raise RuntimeError("Not recognized form argument")

    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2:].astype(float) / 5.0)


def fetch_WS353(which="all"):
    """
    Fetch WS353 dataset for testing attributional and
    relatedness similarity

    Parameters
    ----------
    which : 'all': for both relatedness and attributional similarity,
            'relatedness': for relatedness similarity
            'similarity': for attributional similarity
            'set1': as divided by authors
            'set2': as divided by authors

    References
    ----------
    Finkelstein, Gabrilovich, "Placing Search in Context: The Concept Revisited†", 2002
    Agirre, Eneko et al., "A Study on Similarity and Relatedness Using Distributional and WordNet-based Approaches",
    2009

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)
    """
    if which == "all":
        data = _get_as_pd('https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1',
                          'similarity', header=0, sep="\t")
    elif which == "relatedness":
        data = _get_as_pd('https://www.dropbox.com/s/x94ob9zg0kj67xg/EN-WSR353.txt?dl=1',
                          'similarity', header=None, sep="\t")
    elif which == "similarity":
        data = _get_as_pd('https://www.dropbox.com/s/ohbamierd2kt1kp/EN-WSS353.txt?dl=1',
                          'similarity', header=None, sep="\t")
    elif which == "set1":
        data = _get_as_pd('https://www.dropbox.com/s/opj6uxzh5ov8gha/EN-WS353-SET1.txt?dl=1',
                          'similarity', header=0, sep="\t")
    elif which == "set2":
        data = _get_as_pd('https://www.dropbox.com/s/w03734er70wyt5o/EN-WS353-SET2.txt?dl=1',
                          'similarity', header=0, sep="\t")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(float)

    # We have also scores
    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(float), axis=1).flatten()
        return Bunch(X=X.astype("object"), y=y, sd=sd)
    else:
        return Bunch(X=X.astype("object"), y=y)


def fetch_RG65():
    """
    Fetch Rubenstein and Goodenough dataset for testing attributional and
    relatedness similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)

    References
    ----------
    Rubenstein, Goodenough, "Contextual correlates of synonymy", 1965

    Notes
    -----
    Scores were scaled by factor 10/4
    """
    data = _get_as_pd('https://www.dropbox.com/s/chopke5zqly228d/EN-RG-65.txt?dl=1',
                      'similarity', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(float) * 10.0 / 4.0)


def fetch_RW():
    """
    Fetch Rare Words dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores

    References
    ----------
    Published at http://www-nlp.stanford.edu/~lmthang/morphoNLM/.

    Notes
    -----
    2034 word pairs that are relatively rare with human similarity scores. Rare word selection: our choices of
    rare words (word1) are based on their frequencies – based on five bins (5, 10], (10, 100], (100, 1000],
    (1000, 10000], and the affixes they possess. To create a diverse set of candidates, we randomly
    select 15 words for each configuration (a frequency bin, an affix). At the scale of Wikipedia,
    a word with frequency of 1-5 is most likely a junk word, and even restricted to words with
    frequencies above five, there are still many non-English words. To counter such problems,
    each word selected is required to have a non-zero number of synsets in WordNet(Miller, 1995).
    """
    data = _get_as_pd('https://www.dropbox.com/s/xhimnr51kcla62k/EN-RW.txt?dl=1',
                      'similarity', header=None, sep="\t").values
    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(float),
                 sd=np.std(data[:, 3:].astype(float)))


def fetch_multilingual_SimLex999(which="EN"):
    """
    Fetch Multilingual SimLex999 dataset for testing attributional similarity

    Parameters
    -------
    which : "EN", "RU", "IT" or "DE" for language

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,

    References
    ----------
    Published at http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html.

    Notes
    -----
    Scores for EN are different than the original SimLex999 dataset.

    Authors description:
    Multilingual SimLex999 resource consists of translations of the SimLex999 word similarity data set to
    three languages: German, Italian and Russian. Each of the translated datasets is scored by
    13 human judges (crowdworkers) - all fluent speakers of its language. For consistency, we
    also collected human judgments for the original English corpus according to the same protocol
    applied to the other languages. This dataset allows to explore the impact of the "judgement language"
    (the language in which word pairs are presented to the human judges) on the resulted similarity scores
    and to evaluate vector space models on a truly multilingual setup (i.e. when both the training and the
    test data are multilingual).
    """
    if which == "EN":
        data = _get_as_pd('https://www.dropbox.com/s/nczc4ao6koqq7qm/EN-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "DE":
        data = _get_as_pd('https://www.dropbox.com/s/ucpwrp0ahawsdtf/DE-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "IT":
        data = _get_as_pd('https://www.dropbox.com/s/siqjagyz8dkjb9q/IT-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "RU":
        data = _get_as_pd('https://www.dropbox.com/s/3v26edm9a31klko/RU-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    scores = data.values[:, 2:].astype(float)
    y = np.mean(scores, axis=1)
    sd = np.std(scores, axis=1)

    return Bunch(X=X.astype("object"), y=y, sd=sd)


def fetch_SimLex999(which='all'):
    """
    Fetch SimLex999 dataset for testing attributional similarity
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,
        'conc': matrix with columns conc(w1), conc(w2) and concQ the from dataset
        'POS': vector with POS tag
        'assoc': matrix with columns denoting free association: Assoc(USF) and SimAssoc333
    References
    ----------
    Hill, Felix et al., "Simlex-999: Evaluating semantic models with (genuine) similarity estimation", 2014
    Notes
    -----
     SimLex-999 is a gold standard resource for the evaluation of models that learn the meaning of words and concepts.
     SimLex-999 provides a way of measuring how well models capture similarity, rather than relatedness or
     association. The scores in SimLex-999 therefore differ from other well-known evaluation datasets
     such as WordSim-353 (Finkelstein et al. 2002). The following two example pairs illustrate the
     difference - note that clothes are not similar to closets (different materials, function etc.),
     even though they are very much related: coast - shore 9.00 9.10, clothes - closet 1.96 8.00
    """

    data = _get_as_pd('https://www.dropbox.com/s/0jpa1x8vpmk3ych/EN-SIM999.txt?dl=1',
                      'similarity', sep="\t")

    # We basically select all the columns available
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values

    if which == 'all':
        idx = np.asarray(range(len(X)))
    elif which == '333':
        idx = np.where(assoc[:,1] == 1)[0]
    else:
        raise ValueError("Subset of SL999 not recognized {}".format(which))

    return Bunch(X=X[idx].astype("object"), y=y[idx], sd=sd[idx], conc=conc[idx],
                 POS=POS[idx], assoc=assoc[idx])


def fetch_TR9856():
    """
    Fetch TR9856 dataset for testing multi-word term relatedness

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'topic': vector of topics providing context for each pair of terms

    References
    ----------
    Levy, Ran et al., "TR9856: A multi-word term relatedness benchmark", 2015.

    Notes
    -----
    """
    data = pd.read_csv(os.path.join(_fetch_file(
        'https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_TR9856.v2.zip',
        'similarity', uncompress=True, verbose=0),
        'IBM_Debater_(R)_TR9856.v0.2', 'TermRelatednessResults.csv'), encoding="iso-8859-1")

    # We basically select all the columns available
    X = data[['term1', 'term2']].values
    y = data['score'].values
    topic = data['topic'].values

    return Bunch(X=X.astype("object"), y=y, topic=topic)


def fetch_SimVerb3500(which='all'):
    """
    Fetch SimVerb3500 dataset for testing verb similarity
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
    References
    ----------
    Gerz, Daniela et al., "SimVerb-3500: A Large-Scale Evaluation Set of Verb Similarity", 2016
    Notes
    -----
    TODO
    """
    if which not in ['all', 'dev', 'test']:
        raise RuntimeError("Not recognized which parameter")

    url_map = {"all": 'https://www.dropbox.com/s/xct7j3h7i9bzi7y/all_SimVerb3500.txt?dl=1',
               "dev": 'https://www.dropbox.com/s/57d850d6puxl6nm/dev_SimVerb3500.txt?dl=1',
               "test": 'https://www.dropbox.com/s/66hlkkhfa6c9lrt/test_SimVerb3500.txt?dl=1'}

    data = _get_as_pd(url_map[which], which, header=None, sep=" ")
    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2:].astype(float))




def fetch_SimVerb3500_New(which='all'):
    """
    Fetch SimVerb3500 dataset from local directory and return word similarity pairs.

    Parameters
    ----------
    which : str
        One of ['all', 'dev', 'test'] indicating which split to load.

    Returns
    -------
    data : sklearn.utils.Bunch
        Dictionary-like object with keys:
            'X' : ndarray of shape (n_samples, 2) with word pairs
            'y' : ndarray of shape (n_samples,) with similarity scores
    """

    if which not in ['all', 'dev', 'test']:
        raise RuntimeError("Not recognized 'which' parameter. Use 'all', 'dev', or 'test'.")

    file_map = {
        'all': 'SimVerb-3500.txt',
        'dev': 'SimVerb-500-dev.txt',
        'test': 'SimVerb-3000-test.txt'
    }

    base_path = '/Users/pietro/code/word2gm/evaluation_data/simverb/data'
    full_path = os.path.join(base_path, file_map[which])

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Expected file not found at {full_path}")

    try:
        # It's tab-separated with 6 columns, no header
        data = pd.read_csv(full_path, sep='\t', header=None)
    except Exception as e:
        raise RuntimeError(f"Failed to parse the file: {full_path}\nError: {e}")

    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns (word1, word2, POS, score), got {data.shape[1]} in file: {file_map[which]}")

    return Bunch(
        X=data.iloc[:, 0:2].values.astype("object"),
        y=data.iloc[:, 3].values.astype(float)
    )




def fetch_SCWS():
    """
    Fetch SCWS dataset for testing similarity (with a context)
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with mean scores,
        'sd': standard deviation of scores
    References
    ----------
    Huang et al., "Improving Word Representations via Global Context and Multiple Word Prototypes", 2012
    Notes
    -----
    TODO
    """
    data = _get_as_pd('https://www.dropbox.com/s/qgqj366lzzzj1ua/preproc_SCWS.txt?dl=1', 'similarity', header=None, sep="\t")
    X = data.values[:, 0:2].astype("object")
    mean = data.values[:,2].astype(float)
    sd = np.std(data.values[:, 3:14].astype(float), axis=1).flatten()
    return Bunch(X=X, y=mean,sd=sd)