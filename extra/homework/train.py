import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot
from . import utils

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model = TCN()
    max_len1 = 26
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    data1 = SpeechDataset('data/train.txt',transform = one_hot,max_len = max_len1)
    permutation = torch.randperm(25000)
    #oneh = utils.one_hot(data1)
    #print(oneh.shape)
    print(len(data1))
    #print(len(data1))
    batch_size = 128
    for i in range(0,len(permutation)-batch_size+1,batch_size):
            model.train()
            batch = permutation[i:i+batch_size]
            t_list =[]
            t_label = []
            for j in batch:
                data = data1[j]
                #print(data[:,0:-1].shape)
                #print(data[:,-1].shape)
                #data =img
                t_list.append(data[:,0:-1])
                label = data[:,-1]
                t_label.append(label)
            train_data = torch.stack(t_list)
            #print(train_data.shape)
            train_label = torch.stack(t_label)
            #print(train_label.shape)
            #train_data = train_data.to(device)
            #train_label = train_label.to(device)
            #output = model(train_data)
            string1 = "Hello"
            output1 = model.predict_all(string1)
            print(output1.shape)



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
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
