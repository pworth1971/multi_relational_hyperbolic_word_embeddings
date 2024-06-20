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

    
#
# modified from original to support MPS, Apple silicon
#
class Experiment:

    def __init__(self, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, dataset="wiktionary", device_type='cpu'):
        
        self.model_type = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.device_type = device_type

        # set device type variables
        if device_type == "cuda": 
            self.cuda = True
        else: 
            self.cuda = False

        if device_type == "mps":    
            self.mps = True
        else:
            self.mps = False

        if device_type == "cpu":
            self.cpu = True
        else:
            self.cpu = False     

        self.best_eval_score = 0.0
        self.tasks = {
            'SimVerb3500-dev': fetch_SimVerb3500(which='dev'), 
            'MEN-dev': fetch_MEN(which = "dev"),
        }

        # print params
        print()
        print("-------------------- Experiment Object Params --------------------")
        print(f"model_type: {self.model_type}")
        print(f"dataset: {self.dataset}")
        print(f"learning_rate: {self.learning_rate}")
        print(f"dim: {self.dim}")
        print(f"nneg: {self.nneg}")
        print(f"num_iterations: {self.num_iterations}")
        print(f"batch_size: {self.batch_size}")
        print(f"device_type: {self.device_type}")
        print("cuda: " + str(self.cuda))
        print("mps: " + str(self.mps))
        print("cpu: " + str(self.cpu))
        print("-------------------------------------------------------------")
    
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
        print(f"Training the {self.model_type} multi-relational model...")
        print("device_type: " + str(self.device_type))

        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        #device = "cuda" if self.cuda else "mps" if torch.backends.mps.is_available() else "cpu"
        device = self.device_type                 # set runtime device (Chip Set)    

        train_data_idxs = self.get_data_idxs(d.train_data)
        print(f"Number of training data points: {len(train_data_idxs)}")

        if self.model_type == "poincare":
            model = torch.jit.script(MuRP(d, self.dim))
        else:
            model = torch.jit.script(MuRE(d, self.dim, device=device))
        
        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)

        # note does not appear to support mps, Apple silicon
        if self.cuda:
            model.cuda()
        elif self.cpu:
            model.cpu()

        # Move the model to the appropriate device
        model.to(device)
        
        train_data_idxs_tensor = torch.tensor(train_data_idxs, device=device)
        entity_idxs_lst = list(self.entity_idxs.values())
        negsamples_tbl = torch.tensor(
            np.random.choice(entity_idxs_lst, size=(len(train_data_idxs) // self.batch_size, self.batch_size, self.nneg)),
            device=device
        )

        print()
        print("Starting training...")

        for it in tqdm(range(1, self.num_iterations + 1)):

            print()
            print('--------------------------------------------------------------------------------------')
            print(f"Iteration: {it}")
        
            model.train()
            losses = torch.zeros((len(train_data_idxs) // self.batch_size) + 1, device=device)
            batch_cnt = 0
            train_data_idxs = train_data_idxs_tensor[torch.randperm(train_data_idxs_tensor.shape[0])]

            for j in tqdm(range(0, len(train_data_idxs), self.batch_size)):
                data_batch = train_data_idxs[j:j + self.batch_size]
                negsamples = negsamples_tbl[torch.randint(0, len(train_data_idxs) // self.batch_size, (1,))].squeeze()

                e1_idx = torch.tile(torch.unsqueeze(data_batch[:, 0], 0).T, (1, negsamples.shape[1] + 1))
                r_idx = torch.tile(torch.unsqueeze(data_batch[:, 1], 0).T, (1, negsamples.shape[1] + 1))
                e2_idx = torch.cat((torch.unsqueeze(data_batch[:, 2], 0).T, negsamples[:data_batch.shape[0]]), dim=1)

                targets = torch.zeros(e1_idx.shape, device=device)
                targets[:, 0] = 1

                opt.zero_grad()
                predictions = model(e1_idx, r_idx, e2_idx)
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses[batch_cnt] = loss.detach()
                batch_cnt += 1

            print(f"Loss: {torch.mean(losses).item()}")
            print("______________________________________________________________________________________")

            # Start evaluation
            print("Starting evaluation...")

            model.eval()
            with torch.no_grad():
                # Saving the embeddings as a dictionary
                print("Saving the embeddings for evaluation...")
                embeddings_dict = {}
                for entity in tqdm(self.entity_idxs):
                    if self.model_type == "poincare":
                        embeddings_dict[entity] = model.Eh.weight[self.entity_idxs[entity]].detach().cpu().numpy()
                    else:
                        embeddings_dict[entity] = model.E.weight[self.entity_idxs[entity]].detach().cpu().numpy()

                out_emb_path = os.path.join(settings["output_path"], "embeddings")
                out_model_path = os.path.join(settings["output_path"], "models")
                outfile_path = os.path.join(out_emb_path, f"model_dict_{self.model_type}_{self.dim}_{self.dataset}")
                os.makedirs(out_emb_path, exist_ok=True)
                os.makedirs(out_model_path, exist_ok=True)

                pickle.dump(embeddings_dict, open(outfile_path, "wb"))
                score = self.evaluate(model, outfile_path, self.model_type)
                print(f"Evaluation score: {score}")
                if score > self.best_eval_score:
                    # Saving the embeddings of the best model
                    print("New best model, saving the embeddings...")
                    outfile_path = os.path.join(out_emb_path, f"best_model_dict_{self.model_type}_{self.dim}_{self.dataset}")
                    pickle.dump(embeddings_dict, open(outfile_path, "wb"))
                    self.best_eval_score = score
                    # Saving checkpoint for the best model
                    torch.save({
                        "epoch": it,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": losses,
                    }, os.path.join(out_model_path, f"best_model_checkpoint_{self.model_type}_{self.dim}_{self.dataset}.pt"))


#
# pjw updates:
# updated args list to support multiple devices / chip sets
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiktionary", nargs="?", help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?", help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?", help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=50, nargs="?", help="Learning rate.")
    parser.add_argument("--dim", type=int, default=40, nargs="?", help="Embedding dimensionality.")
    
    #parser.add_argument("--cuda", type=bool, default=False, nargs="?", help="Whether to use CUDA if available or CPU otherwise.")       # deprecated
    #parser.add_argument("--mps", type=bool, default=False, nargs="?", help="Whether to use MPS if available or CPU otherwise.")

    parser.add_argument("--device", type=str, default="cpu", nargs="?", help="MPS, CUDA, or CPU for PyTorch device type.")

    args = parser.parse_args()

     # Print the parsed command line arguments
    print("Command Line Arguments:")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Number of Iterations: {args.num_iterations}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Negative Samples: {args.nneg}")
    print(f"Learning Rate: {args.lr}")
    print(f"Embedding Dimensionality: {args.dim}")
    print(f"Device: {args.device}")

    print()
    print("------ ... executing experiment ... ------")

    dataset = args.dataset
    data_dir = f"data/{dataset}/"

    torch.backends.cudnn.deterministic = True
    
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    d = Data(data_dir=data_dir)

    experiment = Experiment(
        learning_rate=args.lr, 
        batch_size=args.batch_size, 
        num_iterations=args.num_iterations, 
        dim=args.dim, 
        nneg=args.nneg, 
        model=args.model, 
        dataset=args.dataset,
        device_type=args.device
        )

    experiment.train_and_eval()