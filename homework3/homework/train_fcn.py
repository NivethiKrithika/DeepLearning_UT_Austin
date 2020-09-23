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
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0003)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience = 10)
    n_epochs = 11
    train_global_step = 0
    loss = torch.nn.CrossEntropyLoss()
    print(optimizer.param_groups[0]['lr'])
    for iter in range(n_epochs):
        model.train()
        print("epoch is {}".format(iter))
        list_output_train = []
        list_label_train = []
        train_accu = []
        for i,batch in enumerate(train_loader):
            train_data,train_label = batch 
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            output = model(train_data)
            computed_loss = loss(output,train_label.long()).float()
            train_accu.append(accuracy(output,train_label).detach().cpu()
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_global_step +=1
            del(train_data)
            del(train_label)

        print("train accu is {}".format(np.mean(np.array(train_accu))))

        
        model.eval()
        with torch.no_grad():
            list_output_valid = []
            list_label_valid = []
            c = ConfusionMatrix()
            for img, label in valid_loader:
                c.add(model(img.to(device)).argmax(1), label.to(device))
            print("global accuracy is {}".format(c.global_accuracy))
            print("iou is {}".format(c.iou))


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
