import os
import random
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
        self.tasks = {
            'SimVerb3500-dev': fetch_SimVerb3500(which='dev'), 
            'MEN-dev': fetch_MEN(which = "dev"),
        }

        
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


    def train_and_eval(self):
        print("Training the %s multi-relational model..." %self.model_type )

        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        
        device = "cuda" if self.cuda else "cpu"

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model_type == "poincare":
            model = torch.jit.script(MuRP(d, self.dim))
        else:
            model = torch.jit.script(MuRE(d, self.dim))
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
                #Saving the embeddings as a dictionary
                print("Saving the embeddings for evaluation...")
                embeddings_dict = {}
                for entity in tqdm(self.entity_idxs):
                    if self.model_type == "poincare":
                        embeddings_dict[entity] = model.Eh.weight[self.entity_idxs[entity]].detach().cpu().numpy()
                    else:
                        embeddings_dict[entity] = model.E.weight[self.entity_idxs[entity]].detach().cpu().numpy()

                out_emb_path = os.path.join(settings["output_path"], "embeddings")
                out_model_path = os.path.join(settings["output_path"], "models")
                outfile_path = os.path.join(out_emb_path, "model_dict_"+self.model_type+"_" + str(self.dim) + "_" + self.dataset)
                if (not os.path.exists(out_emb_path)):
                    os.makedirs(out_emb_path)
                if (not os.path.exists(out_model_path)):
                    os.makedirs(out_model_path)

                pickle.dump(embeddings_dict, open(outfile_path, "wb"))
                score = self.evaluate(model, outfile_path, self.model_type)
                print("Evaluation score:", score)
                if score > self.best_eval_score:
                    #Saving the embeddings of the best model
                    print("New best model, saving the embeddings...")
                    outfile_path = os.path.join(out_emb_path, "best_model_dict_" + self.model_type + "_" + str(self.dim) + "_" + self.dataset)
                    pickle.dump(embeddings_dict, open(outfile_path, "wb"))
                    self.best_eval_score = score
                    #Saving checkpoint for the best model
                    torch.save({ "epoch": it,
                                 "model_state_dict": model.state_dict(),
                                 "optimizer_state_dict": opt.state_dict(),
                                 "loss": losses,
                        }, os.path.join(out_model_path, "best_model_checkpoint_" + self.model_type + "_" + str(self.dim) + "_" + self.dataset+".pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiktionary", nargs="?",
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
    torch.backends.cudnn.deterministic = True 
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir)
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size, 
                            num_iterations=args.num_iterations, dim=args.dim, 
                            cuda=args.cuda, nneg=args.nneg, model=args.model, dataset = args.dataset)
    experiment.train_and_eval()