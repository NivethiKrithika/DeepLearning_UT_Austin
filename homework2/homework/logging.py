from os import path
import torch
import torch.utils.tensorboard as tb
import numpy

def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    train_global_step = 0
    # This is a strongly simplified training loop
    for epoch in range(20):
        torch.manual_seed(epoch)
        accu = []
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            accu.append(dummy_train_accuracy.mean().item())
            train_logger.add_scalar('loss', dummy_train_loss, global_step = train_global_step)
            train_global_step += 1
            #raise NotImplementedError('Log the training loss')
        train_logger.add_scalar('accuracy',numpy.array(accu).mean(),global_step = train_global_step)
        #raise NotImplementedError('Log the training accuracy')
        torch.manual_seed(epoch)
        accu_valid = []
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            accu_valid.append(dummy_validation_accuracy.mean().item())
        
        valid_logger.add_scalar('accuracy',numpy.array(accu_valid).mean(),global_step = train_global_step)
        #raise NotImplementedError('Log the validation accuracy')



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
