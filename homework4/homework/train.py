import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data,DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb
import os
dir = os.path.dirname(os.path.abspath("__file__"))
dataset_path2 = os.path.join(dir,'dense_data','train')
dataset_path3 = os.path.join(dir,'dense_data','valid')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def train(args):
    sig_layer = torch.nn.Sigmoid()
    from os import path
    model = Detector()
    #train_logger, valid_logger = None, None
    #if args.log_dir is not None:
     #   train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
      #  valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    model = model.to(device)
    #train_logger, valid_logger = None, None
    #if args.log_dir is not None:
     #   train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
      #  valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    #optimizer = torch.optim.Adam(model.parameters(),lr = 3e-3)
    #scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience = 10)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.09,momentum = 0.9,weight_decay = 1e-3)
    n_epochs = 10
    train_global_step = 0
    loss = torch.nn.BCEWithLogitsLoss()
    #print(optimizer.param_groups[0]['lr'])
    dataset = DetectionSuperTuxDataset(dataset_path2,
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor(),
                                                                           dense_transforms.Normalize(mean = [0.485,0.456,0.406],
                                                                                        std = [0.229,0.224,0.225])]))
    batch_size =32
    for iter in range(n_epochs):
        print("iter is {}".format(iter))
        print(optimizer.param_groups[0]['lr'])
        permutation = torch.randperm(9998)
        train_accu = []
        model.train()
        for i in range(0,len(permutation)-batch_size+1,batch_size):
            batch = permutation[i:i+batch_size]
            t_list =[]
            t_label = []
            for j in batch:
                (img,*dets1) = dataset[j]
                data =img
                t_list.append(data)
                #(*dets1 = batch_data[1]
                label,train_size = dense_transforms.detections_to_heatmap(dets1,img.shape[1:])
                t_label.append(label)
            #print(t_list[0].shape)
            train_data = torch.stack(t_list)
            train_label = torch.stack(t_label)
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            output = model(train_data)
            #print(output.shape)
            #print(train_label.shape)
            computed_loss = loss(output,sig_layer(train_label)).float()
            #train_accu.append(accuracy(output,train_label).detach().cpu())
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1
            del(train_data)
            del(train_label)
            print(computed_loss)

        print("train accu is {}".format(np.mean(np.array(train_accu))))

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    raise NotImplementedError('train')
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
