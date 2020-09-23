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
            

        def forward(self, x):
            identity = x
            if(self.down_size):
                identity = self.down_size(x)
            return self.concat_layers(x) + identity

    def __init__(self):
        super().__init__()
        c = 3
        layers = []
        layers.append(torch.nn.Conv2d(c,16,kernel_size = 3,padding = 3,stride = 2))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU())
        c = 16
        L = [64,128,256,512]
        for out_channels in L:
            layers.append(self.Block(c,out_channels,2))
            c = out_channels
        self.final_layers = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(512*3*3,1000),torch.nn.ReLU(),torch.nn.Dropout(p = 0.5))
        self.final = torch.nn.Linear(1000,6)
        
    def forward(self,x):
        z = self.final_layers(x)
        return (self.final(self.classifier(z.view(z.size(0),-1))))


        raise NotImplementedError('CNNClassifier.forward')


class FCN(torch.nn.Module):
    class construct_layer(torch.nn.Module):
        def __init__(self,in_channels,out_channels):
            super().__init__()
            self.concat_layers = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels,3,padding = 1,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
        def forward(self,x): 
            return self.concat_layers(x)


    class up_conv(torch.nn.Module):
        def __init__(self,in_channels,out_channels):
            super().__init__()  
            self.concat_layers1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels,out_channels,3,padding = 1,stride =2,output_padding = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
           
        def forward(self,x): 
            return self.concat_layers1(x)

        #raise NotImplementedError('FCN.__init__')
    def __init__(self):
        super().__init__()
        self.first_conv = self.construct_layer(3,64)
        self.second_conv = self.construct_layer(64,128)
        self.third_conv = self.construct_layer(128,256)
        self.first_up_conv = self.up_conv(256,128)
        self.second_up_conv = self.up_conv(256,64)
        self.third_up_conv = self.up_conv(128,5)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self,x):
        padding_done = 0
        padded_oh = 0
        padded_ow = 0
        if x.size(2) < 16 or x.size(3) < 16:
            padding_done = 1
            ow, oh = x.size(2),x.size(3)
            #print(oh)
            #print(ow)
            padh = 16 - oh if oh < 16 else 0
            padw = 16 - ow if ow < 16 else 0
            padded_ow = ow
            padded_oh = oh
            x = F.pad(x, (0, padh,0, padw), value =0)
        
        first_res = self.first_conv(x)
        max_pool_first = self.pool(first_res)
        second_res =  self.second_conv(max_pool_first)
        max_pool_sec = self.pool(second_res)
        third_res =  self.third_conv(max_pool_sec)       
        max_pool_third = self.pool(third_res)
        first_up_res = self.first_up_conv(max_pool_third)
        second_up_res = self.second_up_conv(torch.cat([first_up_res,max_pool_sec],1))
        final = self.third_up_conv(torch.cat([second_up_res,max_pool_first],1))
        if padding_done == 1:
            final = final[:,:,0:padded_ow,0:padded_oh]
        return final
        
        
        
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
