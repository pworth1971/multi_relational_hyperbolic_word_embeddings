import os
import random
import csv
import numpy as np
import torch
import pickle
import time
from collections import defaultdict
from dynaconf import settings
from definitions_learning.models.multi_relational.load_data import Data
from definitions_learning.models.multi_relational.model import *
from definitions_learning.models.multi_relational.rsgd import *
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity
from web.evaluate import evaluate_similarity_hyperbolic
from web.datasets.similarity import fetch_MEN, fetch_SimVerb3500
from nltk.corpus import stopwords
from tqdm import tqdm
from six import iteritems

import argparse

    
class Experiment:

    def __init__(self, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, dataset="wiktionary", cuda=False):
        self.model_type = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.best_eval_score = 0.0
        self.best_eval_score_avg = 0.0
        self.best_eval_score_supertype = 0.0
        self.best_eval_score_diffquality = 0.0
        self.best_eval_score_diffevent = 0.0
        self.best_eval_score_associatedfact = 0.0
        self.tasks = {
            'SimVerb3500-dev': fetch_SimVerb3500(which='dev'), 
            'MEN-dev': fetch_MEN(which = "dev"),
        }
        oo_vocabulary = []
        for name in self.tasks:    
            for example in self.tasks[name]["X"]:
                oo_vocabulary.append(example[0].lower())
                oo_vocabulary.append(example[1].lower())
        oo_vocabulary = list(set(oo_vocabulary))
        self.oov_definitions_dict = self.load_out_of_vocabulary_definitions(oo_vocabulary, "./data/definitions/cpae/transformer_cpae_DSR_model.csv")
        self.d = Data(data_dir=data_dir, out_of_vocabulary = oo_vocabulary)


    def load_out_of_vocabulary_definitions(self, out_of_vocabulary_list, definitions_path):
        #load definitions
        glosses_data = definitions_path
        #process glosses data
        definitions_dict = {}
        with open(glosses_data, newline='', encoding='ISO-8859-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
            print("Loading definitions for out-of-vocabulary words...")
            for row in tqdm(spamreader):
                #LOAD DICTIONARY
                entry = row[2]
                if not entry in out_of_vocabulary_list:
                    continue
                if not entry in definitions_dict:
                    definitions_dict[entry] = []
                tags = row[4:]
                for tag in tags:
                    if not tag.split("/")[0] in stopwords.words("english"):
                        definitions_dict[entry].append(tag.split("/"))       
        return definitions_dict

        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, embeddings_path, embeddings_type = "poincare"):
            #Load embeddings
            embeddings = load_embedding(embeddings_path, format="dict", normalize=True, lower=True, clean_words=False)
            # Calculate results using helper function
            sp_correlations = []
            for name, examples in iteritems(self.tasks):
                if embeddings_type == "poincare":
                    score = evaluate_similarity_hyperbolic(embeddings, examples.X, examples.y)
                    print(name, score)
                    sp_correlations.append(score)
                else:
                    score = evaluate_similarity(embeddings, examples.X, examples.y)
                    print(name, score)
                    sp_correlations.append(score)
            #return average score
            return np.mean(sp_correlations)

    def test_one_shot(self, model = None):
        #start evaluation
        model.eval()    
        one_shot_embeddings = {}
        one_shot_embeddings_avg = {}
        one_shot_embeddings_supertype = {}
        one_shot_embeddings_diffquality = {}
        one_shot_embeddings_diffevent = {}
        one_shot_embeddings_associatedfact = {}

        for entry in tqdm(self.oov_definitions_dict.keys()):
            e_idx = []
            r_idx = []
            r_idx_supertype = []
            r_idx_diffquality = []
            r_idx_diffevent = []
            r_idx_associatedfact = []
            for definien in self.oov_definitions_dict[entry]:
                if definien[0] in self.entity_idxs and definien[1] in self.relation_idxs:
                    e_idx.append(self.entity_idxs[definien[0]])
                    r_idx.append(self.relation_idxs[definien[1]])
                    if definien[1] == "SUPERTYPE":
                        r_idx_supertype.append(self.relation_idxs[definien[1]])
                    if definien[1] == "DIFFERENTIAQUALITY":
                        r_idx_diffquality.append(self.relation_idxs[definien[1]])
                    if definien[1] == "DIFFERENTIAEVENT":
                        r_idx_diffevent.append(self.relation_idxs[definien[1]])
                    if definien[1] == "ASSOCIATEDFACT":
                        r_idx_associatedfact.append(self.relation_idxs[definien[1]])

            if len(e_idx) == 0 or len(r_idx) == 0:
                continue
            one_shot_embeddings[entry] = model.one_shot_encoding([[e_idx]], [[r_idx]]).detach().cpu().numpy()[0]
            one_shot_embeddings_avg[entry] = model.one_shot_encoding_avg([[e_idx]]).detach().cpu().numpy()[0]
            #supertype
            if len(r_idx_supertype) > 0:
                one_shot_embeddings_supertype[entry] = model.one_shot_encoding([[e_idx]], [[r_idx_supertype]]).detach().cpu().numpy()[0]
            else:
                one_shot_embeddings_supertype[entry] = model.one_shot_encoding_avg([[e_idx]]).detach().cpu().numpy()[0]
            #diffquality
            if len(r_idx_diffquality) > 0:
                one_shot_embeddings_diffquality[entry] = model.one_shot_encoding([[e_idx]], [[r_idx_diffquality]]).detach().cpu().numpy()[0]
            else:
                one_shot_embeddings_diffquality[entry] = model.one_shot_encoding_avg([[e_idx]]).detach().cpu().numpy()[0]
            #diffevent
            if len(r_idx_diffevent) > 0:
                one_shot_embeddings_diffevent[entry] = model.one_shot_encoding([[e_idx]], [[r_idx_diffevent]]).detach().cpu().numpy()[0]
            else:
                one_shot_embeddings_diffevent[entry] = model.one_shot_encoding_avg([[e_idx]]).detach().cpu().numpy()[0]
            #associated fact
            if len(r_idx_associatedfact) > 0:
                one_shot_embeddings_associatedfact[entry] = model.one_shot_encoding([[e_idx]], [[r_idx_associatedfact]]).detach().cpu().numpy()[0]
            else:
                one_shot_embeddings_associatedfact[entry] = model.one_shot_encoding_avg([[e_idx]]).detach().cpu().numpy()[0]
        print("Saving one shot embeddings for evaluation")
        out_emb_path = os.path.join(settings["output_path"], "embeddings")
        out_model_path = os.path.join(settings["output_path"], "models")
        if (not os.path.exists(out_emb_path)):
            os.makedirs(out_emb_path)
        if (not os.path.exists(out_model_path)):
            os.makedirs(out_model_path)
        
        outfile_path = os.path.join(out_emb_path, "model_dict_one_shot_"+self.model_type+"_" + str(self.dim) + "_" + self.dataset)
        pickle.dump(one_shot_embeddings, open(outfile_path, "wb"))
        score = self.evaluate(model, outfile_path, self.model_type)
        
        outfile_path = os.path.join(out_emb_path, "model_dict_one_shot_avg_"+self.model_type+"_" + str(self.dim) + "_" + self.dataset)
        pickle.dump(one_shot_embeddings_avg, open(outfile_path, "wb"))
        score_avg = self.evaluate(model, outfile_path, self.model_type)

        outfile_path = os.path.join(out_emb_path, "model_dict_one_shot_supertype_"+self.model_type+"_"+ str(self.dim) + "_" + self.dataset)
        pickle.dump(one_shot_embeddings_supertype, open(outfile_path, "wb"))
        score_supertype = self.evaluate(model, outfile_path, self.model_type)

        outfile_path = os.path.join(out_emb_path, "model_dict_one_shot_diffquality_"+self.model_type+"_"+ str(self.dim) + "_" + self.dataset)
        pickle.dump(one_shot_embeddings_diffquality, open(outfile_path, "wb"))
        score_diffquality = self.evaluate(model, outfile_path, self.model_type)

        outfile_path = os.path.join(out_emb_path, "model_dict_one_shot_diffevent_"+self.model_type+"_"+ str(self.dim) + "_" + self.dataset)
        pickle.dump(one_shot_embeddings_diffevent, open(outfile_path, "wb"))
        score_diffevent = self.evaluate(model, outfile_path, self.model_type)

        outfile_path = os.path.join(out_emb_path, "model_dict_one_shot_associatedfact_"+self.model_type+"_"+ str(self.dim) + "_" + self.dataset)
        pickle.dump(one_shot_embeddings_associatedfact, open(outfile_path, "wb"))
        score_associatedfact = self.evaluate(model, outfile_path, self.model_type)

        print("=======================================")
        print("Evaluation score:", score)
        if score > self.best_eval_score:
            #Saving the embeddings of the best model
            print("New best model, saving the embeddings...")
            outfile_path = os.path.join(out_emb_path, "best_model_one_shot_dict_" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
            pickle.dump(one_shot_embeddings, open(outfile_path, "wb"))
            self.best_eval_score = score
        print("======================================")
        print("Evaluation score avg:", score_avg)
        if score_avg > self.best_eval_score_avg:
            #Saving the embeddings of the best avg model
            print("New best model avg, saving the embeddings...")
            outfile_path = os.path.join(out_emb_path, "best_model_one_shot_dict_avg_" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
            pickle.dump(one_shot_embeddings_avg, open(outfile_path, "wb"))
            self.best_eval_score_avg = score_avg
        print("======================================")
        print("Evaluation score supertype:", score_supertype)
        if score_supertype > self.best_eval_score_supertype:
            #Saving the embeddings of the best no supertype model
            print("New best model supertype, saving the embeddings...")
            outfile_path = os.path.join(out_emb_path, "best_model_one_shot_dict_supertype" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
            pickle.dump(one_shot_embeddings_supertype, open(outfile_path, "wb"))
            self.best_eval_score_supertype = score_supertype
        print("======================================")
        print("Evaluation score diffquality:", score_diffquality)
        if score_diffquality > self.best_eval_score_diffquality:
            #Saving the embeddings of the best no supertype model
            print("New best model diffquality, saving the embeddings...")
            outfile_path = os.path.join(out_emb_path, "best_model_one_shot_dict_diffquality" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
            pickle.dump(one_shot_embeddings_diffquality, open(outfile_path, "wb"))
            self.best_eval_score_diffquality = score_diffquality
        print("======================================")
        print("Evaluation score diffevent:", score_diffevent)
        if score_diffevent > self.best_eval_score_diffevent:
            #Saving the embeddings of the best no supertype model
            print("New best model diffevent, saving the embeddings...")
            outfile_path = os.path.join(out_emb_path, "best_model_one_shot_dict_diffevent" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
            pickle.dump(one_shot_embeddings_diffevent, open(outfile_path, "wb"))
            self.best_eval_score_diffevent = score_diffevent
        print("======================================")
        print("Evaluation score diffevent:", score_associatedfact)
        if score_associatedfact > self.best_eval_score_associatedfact:
            #Saving the embeddings of the best no supertype model
            print("New best model associatedfact, saving the embeddings...")
            outfile_path = os.path.join(out_emb_path, "best_model_one_shot_dict_associatedfact" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
            pickle.dump(one_shot_embeddings_associatedfact, open(outfile_path, "wb"))
            self.best_eval_score_associatedfact = score_associatedfact


    def train_and_eval(self):
        print("Training the %s multi-relational model..." %self.model_type )
        self.entity_idxs = {self.d.entities[i]:i for i in range(len(self.d.entities))}
        self.relation_idxs = {self.d.relations[i]:i for i in range(len(self.d.relations))}
        device = "cuda" if self.cuda else "cpu"

        train_data_idxs = self.get_data_idxs(self.d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model_type == "poincare":
            model = MuRP(self.d, self.dim)
        else:
            model = MuRE(self.d, self.dim)
        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        
        if self.cuda:
            model.cuda()
            
        train_data_idxs_tensor = torch.tensor(train_data_idxs, device=device)
        entity_idxs_lst = list(self.entity_idxs.values())
        negsamples_tbl = torch.tensor(np.random.choice(entity_idxs_lst, size=(len(train_data_idxs) // self.batch_size, self.batch_size, self.nneg)),
                                      device=device)

        print("Starting training...")
        for it in tqdm(range(1, self.num_iterations+1)):
            model.train()

            losses = torch.zeros((len(train_data_idxs) // self.batch_size) + 1, device=device)
            batch_cnt = 0
            train_data_idxs = train_data_idxs_tensor[torch.randperm(train_data_idxs_tensor.shape[0])]
            for j in tqdm(range(0, len(train_data_idxs), self.batch_size)):
                data_batch = train_data_idxs[j:j+self.batch_size]
                negsamples = negsamples_tbl[torch.randint(0, len(train_data_idxs) // self.batch_size, (1,))].squeeze()
                
                e1_idx = torch.tile(torch.unsqueeze(data_batch[:, 0], 0).T, (1, negsamples.shape[1]+1))
                r_idx = torch.tile(torch.unsqueeze(data_batch[:, 1], 0).T, (1, negsamples.shape[1]+1))
                e2_idx = torch.cat((torch.unsqueeze(data_batch[:, 2], 0).T, negsamples[:data_batch.shape[0]]), dim=1)

                targets = torch.zeros(e1_idx.shape, device=device)
                targets[:, 0] = 1

                opt.zero_grad()

                predictions = model.forward(e1_idx, r_idx, e2_idx)      
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses[batch_cnt] = loss.detach()
                batch_cnt += 1
            print("Iteration:", it)    
            print("Loss:", torch.mean(losses).item())

            #start evaluation
            model.eval()
            with torch.no_grad():
                #evaluate one-shot encoding
                self.test_one_shot(model = model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cpae", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?",
                    help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=50, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dim", type=int, default=40, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    definitions_path = "./data/definitions/wkt/transformer_wkt_DSR_model.csv"
    torch.backends.cudnn.deterministic = True 
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size, 
                            num_iterations=args.num_iterations, dim=args.dim, 
                            cuda=args.cuda, nneg=args.nneg, model=args.model, dataset = args.dataset)
    experiment.train_and_eval()
                

