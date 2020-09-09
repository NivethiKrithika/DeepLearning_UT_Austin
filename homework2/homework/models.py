import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = 3
        layers = []
        layers.append(torch.nn.Conv2d(c,16,kernel_size = 3,padding = 3,stride = 2))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU())
        #layers.append(torch.nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1))
        c = 16
        L = [32,64,128,256]
        for out_channels in L:
            #print("out_channels is {}".format(out_channels))
            #print(c)
            layers.append(torch.nn.Conv2d(c, out_channels,3,padding = 1,stride = 2))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())
            c = out_channels
        self.concat_layers = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(c,6)
        #raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        #y = self.concat_layers(x)
        #print(self.concat_layers(x))
        #print(self.concat_layers(x).mean([2,3]))
        return self.classifier(self.concat_layers(x).mean([2,3]))
        #raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
