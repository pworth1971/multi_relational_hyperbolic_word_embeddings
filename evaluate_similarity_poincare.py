# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_SimVerb3500, fetch_SCWS, fetch_RG65
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity_hyperbolic

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch embedding (warning: it might take few minutes)
fname = "./output/embeddings/best_model_dict_poincare_300_cpae"
embeddings = load_embedding(fname, format="dict", normalize=True, lower=True, clean_words=False)

# Define tasks
tasks = {
    "MEN-dev": fetch_MEN(which="dev"),
    "MEN-test": fetch_MEN(which="test"),
    "SimVerb3500-dev": fetch_SimVerb3500(which="dev"),
    "SimVerb3500-test": fetch_SimVerb3500(which="test"),
    "WS353": fetch_WS353(),
    "WS353-Sim": fetch_WS353(which="similarity"),
    "WS353-Rel": fetch_WS353(which="relatedness"),
    "SimLex999": fetch_SimLex999(),
    "SCWS": fetch_SCWS(),
    "RG": fetch_RG65()
}

# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity_hyperbolic(embeddings, data.X, data.y)))
