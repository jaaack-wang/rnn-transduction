'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Utility functions for training, evaluation,
and deployment (i.e., prediction).
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial

import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import RNN


def init_weights(model, init_method=init.xavier_uniform_):
    '''Initialize model's weights by a given method. Defaults to
        Xavier initialization.'''
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            init_method(param.data)
        else:
            init.constant_(param.data, 0)


def count_parameters(model):
    '''Count the number of trainable parameters.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(ModelConfig, init_method=init.xavier_uniform_):
    '''Customized function to initialze a model given ModelConfig.'''
    in_vocab_size = ModelConfig["in_vocab_size"]
    out_vocab_size = ModelConfig["out_vocab_size"]
    hidden_size = ModelConfig["hidden_size"]
    embd_dim = ModelConfig["embd_dim"]
    num_layers = ModelConfig["num_layers"]
    rnn_type = ModelConfig["rnn_type"]
    bias = ModelConfig["bias"]
    dropout_rate = ModelConfig["dropout_rate"]
    bidirectional = ModelConfig["bidirectional"]
    
    reduction_method = ModelConfig["reduction_method"].lower()
    if reduction_method == "sum":
        reduction_method = torch.sum
    elif reduction_method == "mean":
        reduction_method = torch.mean
    else:
        raise TypeError("reduction_method must be either sum or mean.")

    device = torch.device(ModelConfig["device"])
    model = RNN(in_vocab_size, out_vocab_size, 
            hidden_size, embd_dim, num_layers, 
            rnn_type, device, bias, dropout_rate, 
            bidirectional, reduction_method)

    model.to(device)
    if init_method != None:
        init_weights(model, init_method)
    
    n = count_parameters(model)
    print(f'The model has {n:,} trainable parameters')
    return model


def metrics(Y, Ypred):
    '''Computer the following three metrics:
        - full sequence accuracy: % of sequences correctly generated from end to end
        - first n-symbol accuracy: % of first n symbols correctly generated
        - overlap rate: % of pairwise overlapping symbols
    '''
    # pairwise overlap
    pairwise_overlap = (Y == Ypred).to(torch.float64)
    # pairwise overlap over across sequences (within the given batch)
    per_seq_overlap = pairwise_overlap.mean(dim=0)
    # overlap rate
    overlap_rate = per_seq_overlap.mean().item()
    # full sequence accuracy
    abs_correct = per_seq_overlap.isclose(torch.tensor(1.0, dtype=torch.float64))
    full_seq_accu = abs_correct.to(torch.float64).mean().item()

    # if the n-th symbol does not match, set the following overlapping values to 0
    if pairwise_overlap.dim() <= 1:
        min_idx = pairwise_overlap.argmin(0)
        if pairwise_overlap[min_idx] == 0:
            pairwise_overlap[min_idx:] = 0
        
    else:
        for col_idx, min_idx in enumerate(pairwise_overlap.argmin(0)):
            if pairwise_overlap[min_idx, col_idx] == 0:
                pairwise_overlap[min_idx:, col_idx] = 0
                
    # first n-symbol accuracy 
    first_n_accu = pairwise_overlap.mean().item()
    
    return full_seq_accu, first_n_accu, overlap_rate


def evaluate(model, dataloader, criterion, 
             per_seq_len_performance=False):
    '''Evaluate model performance on a given dataloader.
    "per_seq_len_performance" can be reported if each batch
    in the dataloader only consists of a specific length.
    '''
    model.eval()
    
    if per_seq_len_performance:
        seq_len = set(X.shape[0] for X, _ in dataloader)
        assert len(seq_len) == len(dataloader), "Each batch" \
        " must contain sequences of a specific length. "
        
        perf_log = dict()
        
    # aggragate performance
    aggr_perf = {"loss": 0.0, 
                 "full sequence accuracy": 0.0, 
                 "first n-symbol accuracy": 0.0, 
                 "overlap rate": 0.0}
    
    with torch.no_grad():
        for X, Y in dataloader:
            seq_len, batch_size = Y.shape

            X = X.to(model.device)
            Y = Y.to(model.device)
            logits = model(X)
            
            Ypred = logits.argmax(2)
            full_seq_accu, first_n_accu, overlap_rate = metrics(Y, Ypred)
            loss = criterion(logits.view(seq_len * batch_size, -1), Y.view(-1))
            
            aggr_perf["loss"] += loss.item()
            aggr_perf["full sequence accuracy"] += full_seq_accu
            aggr_perf["first n-symbol accuracy"] += first_n_accu
            aggr_perf["overlap rate"] += overlap_rate
            
            if per_seq_len_performance:
                perf_log[f"Len-{seq_len}"] = {"loss": loss.item(), 
                                                "full sequence accuracy": full_seq_accu, 
                                                "first n-symbol accuracy": first_n_accu, 
                                                "overlap rate": overlap_rate}
            
    aggr_perf = {k:v/len(dataloader) for k,v in aggr_perf.items()}
    
    if per_seq_len_performance:
        perf_log[f"Aggregated"] = aggr_perf
        
        return aggr_perf, perf_log
            
    return aggr_perf

            

def train_loop(model, dataloader, optimizer, criterion):
    '''A single training loop (for am epoch). 
    '''
    model.train()
    
    for X, Y in dataloader:
        seq_len, batch_size = Y.shape
        
        X = X.to(model.device)
        Y = Y.to(model.device)
        optimizer.zero_grad()
        
        logits = model(X)
        
        Ypred = logits.argmax(2)        
        loss = criterion(logits.view(seq_len * batch_size, -1), Y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def train_and_evaluate(model, train_dl, eval_dl, 
                       criterion, optimizer, 
                       saved_model_fp="model.pt", 
                       acc_threshold=0.0, 
                       print_eval_freq=5, 
                       max_epoch_num=10, 
                       train_exit_acc=1.0, 
                       eval_exit_acc=1.0):
    '''Trains and evaluates model while training and returns 
    the training log. The best model with highest full sequence 
    accuracy is saved and returned.
    
    Args:
        - model (nn.Module): a neural network model in PyTorch.
        - train_dl (Dataset): train set dataloader.
        - eval_dl (Dataset): dataloader for evaluation.
        - criterion (method): loss function for computing loss.
        - optimizer (method): Optimization method.
        - saved_model_fp (str): filepath for the saved model (.pt).
        - acc_threshold (float): the min accuracy to save model. 
          Defaults to 0.0. If set greater than 1, no model will be saved.
        - print_eval_freq (int): print and evaluation frequency.
        - max_epoch_num (int): max epoch number. Defaults to 10. Training 
          is stopped if the max epoch number is run out. 
        - train_exit_acc (float): the min train accuracy to exit training.
          Defaults to 1.0. Only takes effect if eval_exit_acc is also met.
        - eval_exit_acc (float): the min eval accu to exit training. Defaults 
          to 1.0. Training is stopped if the eval accuracy if 1.0 or both
          train_exit_acc and eval_exit_acc are met.
    '''
    
    log = dict()
    best_acc, best_epoch = acc_threshold, 0
    epoch, train_acc, eval_acc = 0, 0, 0
    
    while (epoch < max_epoch_num) and (eval_acc != 1.0) and (
        train_acc < train_exit_acc or eval_acc < eval_exit_acc):
        
        epoch += 1
        
        train_loop(model, train_dl, optimizer, criterion)
        
        if epoch % print_eval_freq == 0:
            
            train_perf = evaluate(model, train_dl, criterion)
            train_acc = train_perf['full sequence accuracy']
            
            eval_perf = evaluate(model, eval_dl, criterion)
            eval_acc = eval_perf['full sequence accuracy']
            
            print(f"Current epoch: {epoch}, \ntraining performance: " \
                  f"{train_perf}\nevaluation performance: {eval_perf}\n")
            
            log[f"Epoch#{epoch}"] = {"Train": train_perf, "Eval": eval_perf}
        
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_epoch = epoch
                torch.save(model.state_dict(), saved_model_fp)
    
    if best_acc > acc_threshold:
        log["Best eval accu"] = {"Epoch Number": epoch}
        log["Best eval accu"].update(log[f"Epoch#{best_epoch}"])
        print(saved_model_fp + " saved!\n")
        model.load_state_dict(torch.load(saved_model_fp))
        
    return log


def predict(text, model, dataloader_func, 
            text_decoder, batch_size=1000):
    '''Predict out sequence given input sequence.
    
    Args:
        - model: trained model
        - text (str/list): input sequence(s)
        - dataloader_func (method): customized dataloader function
          that converts text into tensors.
        - text_decoder (method): a method that decodes output tensors
        - batch_size (int): batch size if a        
    '''
    model.eval()
    
    if isinstance(text, str):
        text = [[text, text[0]]]
    elif isinstance(text, list):
        if isinstance(text[0], str):
            text = [[t, t[0]] for t in text]
        elif isinstance(text[0], list):
            text = [[t, t[:1]] for t in text]
            
    dataloader = dataloader_func(text, batch_size=batch_size)
    
    predicted = []
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(model.device)
            logits = model(X)
            Ypred = logits.argmax(2)
            
            for pred in Ypred.transpose(1,0).tolist():
                pred_t = text_decoder(pred)
                predicted.append(pred_t)
            
    return predicted


def customize_predictor(model, dataloader_func, 
                        text_decoder, batch_size=1000):
    '''Customize a predictor function so that the func can be used more easily.'''
    return partial(predict, model=model,
                   dataloader_func=dataloader_func, 
                   text_decoder=text_decoder, 
                   batch_size=batch_size)


def predict_next(next_text, model, in_seq_encoder, out_seq_decoder, re_start=False):
    '''A function that allows predict next output symbol(s) given current input symbol(s).
    
    Args:
        - next_text (str): next input symbol or text.
        - model: trained model
        - in_seq_encoder (method): encodes input sequence into numbers
        - out_seq_decoder (method): decoder output tensors back into str
    '''
    assert isinstance(next_text, str), "next_text must be char or str!"
    
    if re_start:
        model.last_cell = None
        model.last_hidden = None
    
    predicted = []
    for n in in_seq_encoder(next_text):
        x = torch.tensor([n]).long().to(model.device)
        output = model.next_output(x)
        pred = output.argmax(2).item()
        predicted.append(pred)
    
    return out_seq_decoder(predicted)
