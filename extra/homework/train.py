import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot
from . import utils
import random
import os
import numpy as np
def seed_torch(seed=1029):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.enabled = False
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model1 = TCN()
    max_len1 = 26
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    data1 = SpeechDataset('data/train.txt',transform = one_hot)
    valid_data = SpeechDataset('data/valid.txt')
    
    print(len(data1))
    batch_size = 128
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = 20
    optimizer = torch.optim.Adam(model1.parameters(),lr = 6e-5)
    #optimizer = torch.optim.SGD(model1.parameters(),lr = 1e-3,momentum = 0.9,weight_decay = 1e-5)
    for iter in range(n_epochs):
        permutation = torch.randperm(25000)
        j = 0
        for i in range(0,len(permutation)-batch_size+1,batch_size):
            j = j + 1
            model1.train()
            batch = permutation[i:i+batch_size]
            t_list =[]
            t_label = []
            for j in batch:
                data = data1[j]
                t_list.append(data)
                label = data[:,-1]
                t_label.append(label)
            train_data = torch.stack(t_list)
            train_label = torch.stack(t_label)
            output = model1(train_data[:,:,:-1])
            #print(output[:,:,-1].shape)
            #print(train_label.shape)
            loss = criterion(output[:,:,-1],train_label.argmax(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if(j % 10 == 0):
                print("loss is {}".format(loss))
        model1.eval()
        lls = []
        k = 1
        for s in valid_data:
            k = k + 1
            ll = model1.predict_all(s)
            lls.append(float((ll[:, :-1]*utils.one_hot(s)).sum()/len(s)))
            if(k == 1000):
              break
        nll = -np.mean(lls)
        print("nll is {}".format(nll))


    #print(data1[0:2])
    #print(data1[1])
    #print(data1[2])
    
    #str = "Hello"
    #print("Hitting here")
    
    
    #print("output shape is {}".format(output.shape))

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    #raise NotImplementedError('train')
    save_model(model1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
