from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
from .controller import control
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
dir = os.path.dirname(os.path.abspath("__file__"))
dataset_path2 = os.path.join(dir,'drive_data')


def train(args):
    from os import path
    model = Planner()
    model = model.to(device)
    transform2 = dense_transforms.Compose([dense_transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),
                                                        dense_transforms.RandomHorizontalFlip(),
                                                        dense_transforms.ToTensor()])
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer,step_size = 50,gamma = 0.9)

    n_epochs = 200
    
    #dataset = SuperTuxDataset(dataset_path2,
     #                                  transform=transform2)
    train_loader = load_data(dataset_path2,transform = transform2)

    criterion = torch.nn.MSELoss()
    #fl = FocalLoss1()
                                                                       
    batch_size = 128
    
    for iter in range(n_epochs):
        model.train()
        total_loss = []
        print("epoch is {}".format(iter))
        #list_output_train = []
        #list_label_train = []
        for i,batch in enumerate(train_loader):
            train_data,train_label = batch 
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            #list_output_train.append(output)
            #list_label_train.append(train_label)
            output = model(train_data)
            computed_loss = criterion(output,train_label).float()
            total_loss.append(computed_loss)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1

        print("loss is {}".format(np.mean(total_loss).np()))
        model.eval()
        with torch.no_grad():
          if(iter % 5 == 0):
            pytux = PyTux()
            steps1, how_far1 = pytux.rollout('zengarden', control, max_frames=1000,planner = True, verbose=args.verbose)
            print("steps is {}".format(steps1))
            print("how far is {}".format(how_far1))
            pytux.close()            




    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
