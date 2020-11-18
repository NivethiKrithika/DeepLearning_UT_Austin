import torch

from . import utils
from torch.nn.utils import weight_norm
#def one_hot1():
  #one_hot = (torch.randint(len(vocab), (b, 1, length)) == torch.arange(len(vocab))[None, :, None]).float()

class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
    """
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
          super().__init__()
          self.block  = torch.nn.Sequential(torch.nn.ConstantPad1d((2*dilation,0),0),
                                      weight_norm(torch.nn.Conv1d(in_channels,out_channels,kernel_size,dilation = dilation)),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p = 0.1),
                                      torch.nn.ConstantPad1d((2*dilation,0),0),
                                      weight_norm(torch.nn.Conv1d(out_channels,out_channels,kernel_size,dilation = dilation)),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p = 0.1))
          #self.block[1].weight.data.fill_(0.01)
          #K = torch.Tensor([[1 ,0, -1],[2, 0 ,-2], [1, 0 ,-1]])
          #self.block[1].weight = torch.nn.Parameter(0.01)

          self.down_size = None
          self.down_size = weight_norm(torch.nn.Conv1d(in_channels,out_channels,kernel_size = 1))
                                           #      torch.nn.BatchNorm1d(out_channels)) 


          """
          Your code here.
          Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
          Optionally, repeat this pattern a few times and add in a residual block
          :param in_channels: Conv1d parameter
          :param out_channels: Conv1d parameter
          :param kernel_size: Conv1d parameter
          :param dilation: Conv1d parameter
          """
          #raise NotImplementedError('CausalConv1dBlock.__init__')

        def forward(self, x):
            identity = x
            
            if(self.down_size):
                identity = self.down_size(x)
                #print("res size is {}".format(identity.shape))
            return self.block(x)
            raise NotImplementedError('CausalConv1dBlock.forward')

    def __init__(self):
        super().__init__()
        layers = []
        c = 28
        L = [28,32,40,50]
        dilation1 = 1
        for out_channels in L:
            layers.append(self.CausalConv1dBlock(c,out_channels,3,dilation = dilation1))
            c = out_channels
            dilation1 = dilation1 * 2
        self.final_layers = torch.nn.Sequential(*layers)
        self.final_most = torch.nn.Conv1d(c,28,1)
        #self.classifier = torch.nn.Linear(c,28)
        self.soft = torch.nn.Softmax(dim = 1)
        self.m = torch.nn.ConstantPad1d((0, 1), 0)
        self.sig_layer = torch.nn.Sigmoid()
        self.lsoft = torch.nn.LogSoftmax(dim = 1)
        """
        Your code here

        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
        #raise NotImplementedError('TCN.__init__')

    def forward(self, x):
        #print("x shape is {}".format(x.shape))
        t = self.m(x)
        #print("t shape is {}".format(t.shape))
        #print(t.shape)
        z = self.final_layers(t)
        #print(z.shape)
        #z = z.mean([2,3])
        return (self.soft(self.final_most(z)))
        """
        Your code here
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        raise NotImplementedError('TCN.forward')

    def predict_all(self, some_text):
        #print("some_txt is {}".format(some_text))
        oneh = utils.one_hot(some_text)
        #print("input shape is {}".format(oneh.shape))
        t = self.m(oneh[None,:,:])
        #print("t shape is {}".format(t.shape))
        #print(t.shape)
        z = self.lsoft(self.final_most(self.final_layers(t)))
        #print("final shape is {}".format(z.shape))
        #print("returned shape is  {}".format(torch.squeeze(z,0).shape))
        #print(torch.squeeze(z).shape)
        return(torch.squeeze(z,0))
        """
        Your code here

        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        #raise NotImplementedError('TCN.predict_all')


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
