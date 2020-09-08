from .models import ClassificationLoss, model_factory, save_model, LinearClassifier, MLPClassifier
from .utils import accuracy, load_data
import torch
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dir = os.path.dirname(os.path.abspath("__file__"))
#print(dir)


dataset_path1 = os.path.join(dir, 'data','train')
dataset_path2 = os.path.join(dir, 'data','valid')

def train(args):
    model = model_factory[args.model]()
    if(args.model == 'mlp'):
        model = MLPClassifier().to(device)
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.004,momentum = 0.9,weight_decay = 1e-3)
        train_loader = load_data(dataset_path1,batch_size = 512)
        valid_loader = load_data(dataset_path2,batch_size = 512)
        n_epochs = 40
    else:
        model = LinearClassifier().to(device)
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9,weight_decay = 5*1e-3)
        train_loader = load_data(dataset_path1,batch_size = 512)
        valid_loader = load_data(dataset_path2,batch_size = 512)
        n_epochs = 50
    train_global_step = 0
    for iter in range(n_epochs):
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
            
            loss = ClassificationLoss()
            computed_loss = loss(output,train_label)
            #train_logger.add_scalar('train/loss',computed_loss,global_step = train_global_step)
            #print("train loss is {} ".format(computed_loss))
           
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1
        aggregated_output = torch.cat(list_output_train)
        aggregated_label = torch.cat(list_label_train)
        train_accu = accuracy(aggregated_output,aggregated_label)
        #train_logger.add_scalar('train/accu',train_accu,global_step = train_global_step)
        print("train accu is {}".format(train_accu))
        
        with torch.no_grad():
            list_output_valid = []
            list_label_valid = []
            for i, valid_batch in enumerate(valid_loader):
                valid_data,valid_label = valid_batch
                valid_data,valid_label = valid_data.to(device), valid_label.to(device)
                valid_output = model(valid_data)
                valid_loss = ClassificationLoss()
                computed_valid_loss = valid_loss(valid_output,valid_label)
                #train_logger.add_scalar('valid/loss',computed_valid_loss,global_step = train_global_step)
                #print("valid loss is {}".format(computed_valid_loss))
                list_output_valid.append(valid_output)
                list_label_valid.append(valid_label)

            aggregated_valid_output = torch.cat(list_output_valid)
            aggregated_valid_label = torch.cat(list_label_valid)
            accu = accuracy(aggregated_valid_output,aggregated_valid_label) 
            #train_logger.add_scalar('valid/accu',accu,global_step = train_global_step)
            print("valid accu is {}".format(accu))
            

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
