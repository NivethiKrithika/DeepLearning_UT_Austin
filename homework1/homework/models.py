import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        loss = torch.nn.CrossEntropyLoss()
        #print(input.shape)
        #print(target.shape)
        computed_loss = loss(input,target.long())
        return computed_loss
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        raise NotImplementedError('ClassificationLoss.forward')

input_size = 64*64*3
output_size = 6

class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size,50)
        self.linear2 = torch.nn.Linear(50,100)
        self.linear3 = torch.nn.Linear(100,output_size)

    def forward(self, x):
        return self.linear3(self.linear2(self.linear1(x.view(x.size(0),-1))))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear1 = torch.nn.Linear(64*64*3,50)
        relu1 = torch.nn.ReLU()
       # linear2 = torch.nn.Linear(50,100)
        #relu2 = torch.nn.ReLU()
        linear3 = torch.nn.Linear(50,100)
        relu3 = torch.nn.ReLU()
        linear4 = torch.nn.Linear(100,6)
        self.final = torch.nn.Sequential(linear1,relu1,linear3,relu3,linear4)

    def forward(self, x):
        return self.final(x.view(x.shape[0],-1))
        #raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))



def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
