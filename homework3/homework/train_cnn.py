from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, accuracy,LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import os
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dir = os.path.dirname(os.path.abspath("__file__"))
print(dir)
dataset_path1 = os.path.join(dir, 'data','train')
dataset_path2 = os.path.join(dfrom .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, accuracy,LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import os
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dir = os.path.dirname(os.path.abspath("__file__"))
print(dir)
dataset_path1 = os.path.join(dir, 'data','train')
dataset_path2 = os.path.join(dir, 'data','valid')
import numpy as np

def do_transform(horizontalFlip = False,randomCrop = None, colourjitter = False,resize = None ):
    transforms = []
    if(horizontalFlip):
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    if(randomCrop is not None):
        transforms.append(torchvision.transforms.RandomResizedCrop(randomCrop))
    if (resize is not None):
        transforms.append(torchvision.transforms.Resize(resize))
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    validation_accuracies = []
    transforms = do_transform(horizontalFlip =True,randomCrop =None,colourjitter = False,resize = None)
    train_loader = load_data(dataset_path1)
    valid_loader = load_data(dataset_path2)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum = 0.9,weight_decay = 1e-3)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience = 50)
    n_epochs = 30
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
            computed_loss = loss(output,train_label.long()).float()
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1
        aggregated_output = torch.cat(list_output_train)
        aggregated_label = torch.cat(list_label_train)
        train_accu = accuracy(aggregated_output,aggregated_label)
        print("train accu is {}".format(train_accu))
        
        model.eval()
        with torch.no_grad():
            list_output_valid = []
            list_label_valid = []
            for i, valid_batch in enumerate(valid_loader):
                valid_data,valid_label = valid_batch
                valid_data,valid_label = valid_data.to(device), valid_label.to(device)
                valid_output = model(valid_data)
                list_output_valid.append(valid_output)
                list_label_valid.append(valid_label)

            aggregated_valid_output = torch.cat(list_output_valid)
            aggregated_valid_label = torch.cat(list_label_valid)
            accu = accuracy(aggregated_valid_output,aggregated_valid_label)
            print("validation accu is {}".format(accu)) 
            if(accu > 0.92):
              break
            validation_accuracies.append(accu)
            scheduler.step(np.mean(np.array(validation_accuracies),dtype = np.float))
            #valid_logger.add_scalar('accuracy',accu,global_step = train_global_step)
            #print("valid accu is {}".format(accu))


    """
    Your code here, modify your HW1 code
    
    """

    save_model(model)


    """
    Your code here, modify your HW1 / HW2 code
    """
    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

ir, 'data','valid')

def do_transform(horizontalFlip = False,randomCrop = None, colourjitter = False,resize = None ):
    transforms = []
    if(horizontalFlip):
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    if(randomCrop is not None):
        transforms.append(torchvision.transforms.RandomResizedCrop(randomCrop))
    if (resize is not None):
        transforms.append(torchvision.transforms.Resize(resize))
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
     #   from os import path
    #model = CNNClassifier()
    validation_accuracies = []
    transforms = do_transform(horizontalFlip =True,randomCrop =None,colourjitter = False,resize = None)
    train_loader = load_data(dataset_path1)
    valid_loader = load_data(dataset_path2)
    #dataset = SuperTuxDataset(dataset_path,transform = transforms)
    #train_logger, valid_logger = None, None
    #if args.log_dir is not None:
     #   train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
      #  valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    model = model.to(device)
    #train_logger, valid_logger = None, None
    #if args.log_dir is not None:
     #   train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
      #  valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum = 0.9,weight_decay = 1e-3)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience = 50)
    n_epochs = 30
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
            computed_loss = loss(output,train_label.long()).float()

           
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
        #train_logger.add_scalar('accuracy',train_accu,global_step = train_global_step)
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
                #valid_logger.add_scalar('loss',computed_valid_loss,global_step = train_global_step)
                #print("valid loss is {}".format(computed_valid_loss))
                list_output_valid.append(valid_output)
                list_label_valid.append(valid_label)

            aggregated_valid_output = torch.cat(list_output_valid)
            aggregated_valid_label = torch.cat(list_label_valid)
            accu = accuracy(aggregated_valid_output,aggregated_valid_label) 
            validation_accuracies.append(accu)
            scheduler.step(np.mean(np.array(validation_accuracies),dtype = np.float))
            #valid_logger.add_scalar('accuracy',accu,global_step = train_global_step)
            print("valid accu is {}".format(accu))


    """
    Your code here, modify your HW1 code
    
    """

    save_model(model)


    """
    Your code here, modify your HW1 / HW2 code
    """
    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

