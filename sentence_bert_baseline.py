import os
import random
import numpy as np
import torch
import pickle
import time
import csv
from dynaconf import settings
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_SimVerb3500, fetch_SCWS, fetch_RG65
from tqdm import tqdm
from six import iteritems
from sentence_transformers import SentenceTransformer

import argparse

    
class Experiment:

    def __init__(self, model_name="bert-base-uncased", definition_path="./data/definitions/cpae/cpae_definitions.csv", cuda=True):
        self.model_name = model_name
        self.dataset = dataset
        self.cuda = cuda
        self.tasks = {
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
        #load pretrained model
        self.model = SentenceTransformer(model_name)
        #load definitions (definenda, definitions)
        self.definitions = self.load_definitions_from_csv(definition_path)
        #encode definitions
        sentences = self.definitions[1]
        #definitions are encoded by calling model.encode()
        embeddings = self.model.encode(sentences, show_progress_bar = True)
        #create embeddings dictionary
        embeddings_dict = {}
        index = 0
        print("Saving the embeddings for evaluation...")
        for definendum in self.definitions[0]:
            embeddings_dict[definendum] = embeddings[index]
            index += 1
        #save embeddings for evaluation
        out_emb_path = os.path.join(settings["output_path"], "embeddings")
        outfile_path = os.path.join(out_emb_path, "model_dict_"+self.model_name)
        if (not os.path.exists(out_emb_path)):
            os.makedirs(out_emb_path)
        pickle.dump(embeddings_dict, open(outfile_path, "wb"))
        score = self.evaluate(outfile_path)
        print("Avg score: ", score)

        
    def load_definitions_from_csv(self, data):
        definitions = {}
        with open(data, newline='', encoding='ISO-8859-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
            for row in spamreader:
                if not row[2] in definitions:
                    definitions[row[2]] = row[3]
                else:
                    definitions[row[2]] += " " + row[3]
        return list(definitions.keys()), list(definitions.values())

    def evaluate(self, embeddings_path):
        #Load embeddings
        embeddings = load_embedding(embeddings_path, format="dict", normalize=True, lower=True, clean_words=False)
        # Calculate results using helper function
        sp_correlations = []
        for name, examples in iteritems(self.tasks):
                score = evaluate_similarity(embeddings, examples.X, examples.y)
                print(name, score)
                sp_correlations.append(score)
        #return average score
        return np.mean(sp_correlations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cpae", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
    parser.add_argument("--dim", type=int, default=300, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(model_name = "sentence-t5-large")
                

