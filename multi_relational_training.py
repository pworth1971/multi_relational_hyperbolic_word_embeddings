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

from web.datasets.similarity import fetch_MEN, fetch_SimVerb3500_New

from tqdm import tqdm
from six import iteritems

import argparse


#
# pjw additions
#
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#
# modified from original to support MPS, Apple silicon
#
class Experiment:

    def __init__(self, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, dataset="wiktionary", device_type='cpu', early_stopping=3):
        
        self.model_type = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.device_type = device_type
        self.early_stopping = early_stopping

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
#            'SimVerb3500-dev': fetch_SimVerb3500(which='dev'),
            'SimVerb3500-dev': fetch_SimVerb3500_New(which='dev'), 
            'MEN-dev': fetch_MEN(which = "dev"),
        }

        print(time.ctime())

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
        print("early_stopping: " + str(self.early_stopping))
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
        
        return np.mean(sp_correlations)         #return average score



    # ------------------------------------------------------------------------------------------------
    # Experiment::train_and_eval():
    #
    # train and evaluate the model
    #  
    def train_and_eval(self):

        print(f"Training the {self.model_type} multi-relational model...")
        print("device_type: " + str(self.device_type))

        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        device = self.device_type                 # set runtime device (Chip Set)    

        # print CUDA info if available
        if torch.cuda.is_available():
            print()
            print('----------------------- CUDA INFO -----------------------')
            print("CUDA Available:", torch.cuda.is_available())
            print("CUDA Version:", torch.version.cuda)
            print("Number of GPUs:", torch.cuda.device_count())
            print("Current Device:", torch.cuda.current_device())
            print('----------------------------------------------------------')
            print()
    
        train_data_idxs = self.get_data_idxs(d.train_data)
        print(f"Number of training data points: {len(train_data_idxs)}")

        # instantiate the pytorch jit model
        if self.model_type == "poincare":
            model = torch.jit.script(MuRP(d, self.dim))
        else:
            model = torch.jit.script(MuRE(d, self.dim))
        
        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)

        # Move the model to GPU as available
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
            model.cuda()
        else:
            model.to(device)

        train_data_idxs_tensor = torch.tensor(train_data_idxs, device=device)
        entity_idxs_lst = list(self.entity_idxs.values())

        negsamples_tbl = torch.tensor(
            np.random.choice(entity_idxs_lst, size=(len(train_data_idxs) // self.batch_size, self.batch_size, self.nneg)),
            device=device
        )

        self.loss_history = []  # Initialize loss history

        print()
        print("Starting training...")

        for it in tqdm(range(1, self.num_iterations + 1)):
            
            start_time = time.ctime()
            print(start_time)

            print()
            print('--------------------------------------------------------------------------------------')
            print(f"***** Iteration: {it} *****")
        
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

                if torch.cuda.is_available():
                    loss = model.module.loss(predictions, targets)
                else:
                    loss = model.loss(predictions, targets)
                
                loss.backward()
                opt.step()
                losses[batch_cnt] = loss.detach()
                batch_cnt += 1

            mean_loss = torch.mean(losses).item()
            self.loss_history.append(mean_loss)                 # Store mean loss per iteration
            print(f"Loss: {mean_loss}")

            #
            # Start evaluation
            #
            model.eval()
            with torch.no_grad():

                # Saving the embeddings as a dictionary
                print("Saving the embeddings for evaluation...")
                embeddings_dict = {}
                
                for entity in tqdm(self.entity_idxs):

                    if self.model_type == "poincare":
                        if torch.cuda.is_available():
                            embeddings_dict[entity] = model.module.Eh.weight[self.entity_idxs[entity]].detach().cpu().numpy()
                        else:
                            embeddings_dict[entity] = model.Eh.weight[self.entity_idxs[entity]].detach().cpu().numpy()
                    else:
                        if torch.cuda.is_available():
                            embeddings_dict[entity] = model.module.E.weight[self.entity_idxs[entity]].detach().cpu().numpy()
                        else:
                            embeddings_dict[entity] = model.E.weight[self.entity_idxs[entity]].detach().cpu().numpy()

                out_emb_path = os.path.join(settings["output_path"], "embeddings")
                out_model_path = os.path.join(settings["output_path"], "models")
                outfile_path = os.path.join(out_emb_path, f"model_dict_{self.model_type}_{self.dim}_{self.dataset}")
                os.makedirs(out_emb_path, exist_ok=True)
                os.makedirs(out_model_path, exist_ok=True)

                # Save .pkl
                pickle.dump(embeddings_dict, open(outfile_path, "wb"))

                # Save .vec
                vec_file = outfile_path + ".vec"
                with open(vec_file, "w") as fvec:
                    dim = len(next(iter(embeddings_dict.values())))
                    fvec.write(f"{len(embeddings_dict)} {dim}\n")
                    for word, vector in embeddings_dict.items():
                        vector_str = " ".join([f"{x:.6f}" for x in vector])
                        fvec.write(f"{word} {vector_str}\n")

                score = self.evaluate(model, outfile_path, self.model_type)
                print(f"Evaluation score: {score}")
                
                # Early stopping logic
                improvement = (score - self.best_eval_score) / max(self.best_eval_score, 1e-8) * 100
                if improvement < 0.01:
                    self.no_improvement_count += 1
                    print(f"No significant improvement ({improvement:.5f}%), count = {self.no_improvement_count}")
                else:
                    self.no_improvement_count = 0

                if self.no_improvement_count >= self.early_stopping:
                    print(f'Early stopping triggered: no significant improvement in last {self.early_stopping} evaluations.')
                    break

                #
                # if the model is the best, save it and store info
                #
                if score > self.best_eval_score:                                           
                    
                    print("New best model, saving the embeddings...")
                    best_outfile_path = os.path.join(out_emb_path, f"best_model_dict_{self.model_type}_{self.dim}_{self.dataset}")

                    # Save best .pkl
                    pickle.dump(embeddings_dict, open(best_outfile_path, "wb"))
                    
                    # Save best .vec
                    best_vec_file = best_outfile_path + ".vec"
                    with open(best_vec_file, "w") as fvec:
                        dim = len(next(iter(embeddings_dict.values())))
                        fvec.write(f"{len(embeddings_dict)} {dim}\n")
                        for word, vector in embeddings_dict.items():
                            vector_str = " ".join([f"{x:.6f}" for x in vector])
                            fvec.write(f"{word} {vector_str}\n")
                            
                    self.best_eval_score = score
                    
                    # Saving checkpoint for the best model
                    torch.save({
                        "epoch": it,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": losses,
                    }, os.path.join(out_model_path, f"best_model_checkpoint_{self.model_type}_{self.dim}_{self.dataset}.pt"))

        iteration_end = time.time()

        end_time = time.ctime()
        print(end_time)

        # Plotting after training
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss Per Iteration')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig('output/plot.png')

        plt.show()

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
    
    args = parser.parse_args()

    # Auto-select the device
    if torch.cuda.is_available():
        selected_device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        selected_device = "mps"
    else:
        selected_device = "cpu"

    args.device = selected_device  # override args.device
    
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

    print()
    print('----------------------- Environment variables -----------------------')
    print(os.environ)
    print('---------------------------------------------------------------------')
    print()
    
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