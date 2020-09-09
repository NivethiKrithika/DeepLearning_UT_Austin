from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import os
dir = os.path.dirname(os.path.abspath("__file__"))
dataset_path1 = os.path.join(dir, 'data','train')
dataset_path2 = os.path.join(dir, 'data','valid')
train_loader = load_data(dataset_path1,batch_size = 32)
valid_loader = load_data(dataset_path2,batch_size = 32)


def train(args):
    from os import path
    model = CNNClassifier()
    model = model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum = 0.9,weight_decay = 1e-3)
    n_epochs = 10
    train_global_step = 0
    loss = torch.nn.CrossEntropyLoss()
    for iter in range(n_epochs):
        model.train()
        print("epoch is {}".format(iter))
        list_output_train = []
        list_label_train = []
        for i,batch in enumerate(train_loader):
            train_data,train_label = batch
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            output = model(train_data)
            list_output_train.append(output)
            list_label_train.append(train_label)
            #print(train_data.shape)
            #print(train_data)
            #print("output shape is")
            #print(output.shape)
            #if(iter > 18):
              #print("output is")
              #print(output)
            
            computed_loss = loss(output,train_label.long()).float()
            #print("loss type is {}".format(computed_loss.dtype))
            #print(computed_loss)
            train_logger.add_scalar('loss',computed_loss,global_step = train_global_step)
            #print("train loss is {} ".format(computed_loss))
           
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1
        aggregated_output = torch.cat(list_output_train)
        #print("aggregated_output is")
        #print(aggregated_output)
        aggregated_label = torch.cat(list_label_train)
        #print("aggregated_label is")
        #print(aggregated_label)
        train_accu = accuracy(aggregated_output,aggregated_label)
        train_logger.add_scalar('accuracy',train_accu,global_step = train_global_step)
        print("train accu is {}".format(train_accu))
        
        model.eval()
        with torch.no_grad():
            list_output_valid = []
            list_label_valid = []
            for i, valid_batch in enumerate(valid_loader):
                valid_data,valid_label = valid_batch
                valid_data,valid_label = valid_data.to(device), valid_label.to(device)
                valid_output = model(valid_data)
                computed_valid_loss = loss(valid_output,valid_label.long())
                valid_logger.add_scalar('loss',computed_valid_loss,global_step = train_global_step)
                #print("valid loss is {}".format(computed_valid_loss))
                list_output_valid.append(valid_output)
                list_label_valid.append(valid_label)

            aggregated_valid_output = torch.cat(list_output_valid)
            aggregated_valid_label = torch.cat(list_label_valid)
            accu = accuracy(aggregated_valid_output,aggregated_valid_label) 
            valid_logger.add_scalar('accuracy',accu,global_step = train_global_step)
            print("valid accu is {}".format(accu))

    """
    Your code here, modify your HW1 code
    
    """

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
