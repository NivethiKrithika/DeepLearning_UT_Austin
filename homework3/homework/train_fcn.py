import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, accuracy, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
dir = os.path.dirname(os.path.abspath("__file__"))
dataset_path2 = os.path.join(dir,'dense_data','train')
dataset_path3 = os.path.join(dir,'dense_data','valid')

def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    validation_accuracies = []
    #transforms = do_transform(horizontalFlip =True,randomCrop =None,colourjitter = False,resize = None)
    train_loader = load_dense_data(dataset_path2)
    valid_loader = load_dense_data(dataset_path3)
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
            #output1 = torch.argmax(output,dim = 1)

            #print(train_data.shape)
            #print(train_data)
            #print("output shape is")
            #print(output.shape)
            #if(iter > 18):
              #print("output is")
              #print(output)
            #print(output1.shape)
            #print(train_label.shape)
            computed_loss = loss(output,train_label.long()).float()
            #print("loss type is {}".format(computed_loss.dtype))
            #print(computed_loss)
            #train_logger.add_scalar('loss',computed_loss,global_step = train_global_step)
            #print("train loss is {} ".format(computed_loss))
           
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1
            list_output_train.append(output)
            list_label_train.append(train_label)
        aggregated_output = torch.cat(list_output_train).detach().cpu()
        #print("aggregated_output is")
        #print(aggregated_output)
        aggregated_label = torch.cat(list_label_train)
        del(list_output_train)
        del(list_label_train)
        #print("aggregated_label is")
        #print(aggregated_label)
        train_accu = accuracy(aggregated_output,aggregated_label).float()
        #train_logger.add_scalar('accuracy',train_accu,global_step = train_global_step)
        print("train accu is {}".format(train_accu))
        del(aggregated_output)
        del(aggregated_label)
        
        model.eval()
        with torch.no_grad():
            list_output_valid = []
            list_label_valid = []
            for i, valid_batch in enumerate(valid_loader):
                valid_data,valid_label = valid_batch
                valid_data,valid_label = valid_data.to(device), valid_label.to(device)
                valid_output = model(valid_data)
                #valid_output1 = torch.argmax(valid_output,dim = 1)
                computed_valid_loss = loss(valid_output,valid_label.long()).float()
                #valid_logger.add_scalar('loss',computed_valid_loss,global_step = train_global_step)
                #print("valid loss is {}".format(computed_valid_loss))
                list_output_valid.append(valid_output)
                list_label_valid.append(valid_label)

            aggregated_valid_output = torch.cat(list_output_valid)
            aggregated_valid_label = torch.cat(list_label_valid)
            accu = accuracy(aggregated_valid_output,aggregated_valid_label).float().detach() 
            validation_accuracies.append(accu)
            scheduler.step(np.mean(np.array(validation_accuracies),dtype = np.float))
            #valid_logger.add_scalar('accuracy',accu,global_step = train_global_step)
            print("valid accu is {}".format(accu))

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
