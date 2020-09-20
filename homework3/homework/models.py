import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self,in_channels,out_channels,stride):
            super().__init__()
            c = in_channels
            self.concat_layers = torch.nn.Sequential(torch.nn.Conv2d(c, out_channels,3,padding = 1,stride = stride),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Dropout(p = 0.1),
                                                     torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
            
            self.down_size = None
            self.down_size = torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = stride),
                                                 torch.nn.BatchNorm2d(out_channels)) 
            
            #if(stride != 1):
         
            
            #self.concat_layers = torch.nn.Sequential(*layers)
          #  self.classifier = torch.nn.Linear(c,6)
        #raise NotImplementedError('CNNClassifier.__init__')

        def forward(self, x):
            identity = x
            if(self.down_size):
                identity = self.down_size(x)
            return self.concat_layers(x) + identity
                 
            #y = self.concat_layers(x)
            #print(self.concat_layers(x))
            #print(self.concat_layers(x).mean([2,3]))
            #return self.classifier(self.concat_layers(x).mean([2,3]))
            #raise NotImplementedError('CNNClassifier.forward')
    def __init__(self):
        super().__init__()
        c = 3
        layers = []
        layers.append(torch.nn.Conv2d(c,16,kernel_size = 3,padding = 3,stride = 2))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU())
        c = 16
        L = [32,64,128,256]
        for out_channels in L:
            layers.append(self.Block(c,out_channels,2))
            c = out_channels
        self.final_layers = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(256*3*3,1000),torch.nn.ReLU(),torch.nn.Dropout(p = 0.5))
        self.final = torch.nn.Linear(1000,6)
        
    def forward(self,x):
        z = self.final_layers(x)
        #z = z.mean([2,3])
        return (self.final(self.classifier(z.view(z.size(0),-1))))
            #print("out_channels is {}".format(out_channels))
            #print(c)


        raise NotImplementedError('CNNClassifier.forward')


class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self,in_channels,out_channels,stride):
            super().__init__()
            c = in_channels
            self.concat_layers = torch.nn.Sequential(torch.nn.Conv2d(c, out_channels,3,padding = 1,stride = stride),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
                                                     #torch.nn.Dropout(p = 0.1),
                                                     #torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1),
                                                     #torch.nn.BatchNorm2d(out_channels),
                                                     #torch.nn.ReLU())
            
            self.down_size = None
            self.down_size = torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = stride),
                                                 torch.nn.BatchNorm2d(out_channels)) 
            
            #if(stride != 1):
         
            
            #self.concat_layers = torch.nn.Sequential(*layers)
          #  self.classifier = torch.nn.Linear(c,6)
        #raise NotImplementedError('CNNClassifier.__init__')

        def forward(self, x):
            identity = x
            if(self.down_size):
                identity = self.down_size(x)
            return self.concat_layers(x) + identity
    class Block1(torch.nn.Module):
        def __init__(self,in_channels,out_channels,stride):
            super().__init__()
            c = in_channels
            self.concat_layers1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(c, out_channels,3,padding = 1,stride = 2),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
                                                     #torch.nn.Dropout(p = 0.1),
                                                     #torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1),
                                                     #torch.nn.BatchNorm2d(out_channels),
                                                     #torch.nn.ReLU())
            
            #self.down_size = None
            #self.down_size = torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = stride),
                                                # torch.nn.BatchNorm2d(out_channels)) 
            
            #if(stride != 1):
         
            
            #self.concat_layers = torch.nn.Sequential(*layers)
          #  self.classifier = torch.nn.Linear(c,6)
        #raise NotImplementedError('CNNClassifier.__init__')

        def forward(self, x):
            #identity = x
            #if(self.down_size):
             #   identity = self.down_size(x)
            m=self.concat_layers1(x)
            #print(m.shape)
            return(R.pad(m, pad=(0,1,0,1), mode='constant', value=0))
            #print(m.shape)
            #return m
            
                 
            #y = self.concat_layers(x)
            #print(self.concat_layers(x))
            #print(self.concat_layers(x).mean([2,3]))
            #return self.classifier(self.concat_layers(x).mean([2,3]))
            #raise NotImplementedError('CNNClassifier.forward')
    def __init__(self):
        super().__init__()
        c = 3
        layers = []
        #layers.append(torch.nn.Conv2d(c,16,kernel_size = 3,padding = 3,stride = 2))
        #layers.append(torch.nn.BatchNorm2d(16))
        #layers.append(torch.nn.ReLU())
        #c = 
        L = [32,64,128]
        for out_channels in L:
            layers.append(self.Block(c,out_channels,1))
            layers.append(torch.nn.MaxPool2d(2))
            c = out_channels
        self.final_layers = torch.nn.Sequential(*layers)
        
        self.up_conv_layer = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                                         torch.nn.BatchNorm2d(128),
                                                         torch.nn.ReLU())
        layers1 = []
        n = 128
        M = [64,32,32]
        for out_channels1 in M: 
            layers1.append(self.Block1(n,out_channels1,2))
            n = out_channels1
        self.final = torch.nn.Sequential(*layers1)
        self.out_conv = torch.nn.Conv2d(32,6,1)
                        #torch.nn.ConvTranspose2d(128,64,3,stride = 2, padding = 1),
                        #torch.nn.pad(img, (0, 0, padw, padh), fill=fill)
                        #torch.nn.ConvTranspose2d(64,32,2,stride = 2, padding = 1),
                        #torch.nn.ConvTranspose2d(32,3,2,stride = 2, padding = 1))
        #self.classifier = torch.nn.Linear(c,6)
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        #raise NotImplementedError('FCN.__init__')

        
    def forward(self,x):
        z = self.final_layers(x)
        #print(z.shape)
        m = self.up_conv_layer(z)
        #print(m.shape) 
        n = self.final(m)
        #print(n.shape)
        #z = z.mean([2,3])
        q =self.out_conv(n)
        #print(q.shape)
        return q
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
