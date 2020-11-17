from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch
char_set =['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ','.']
def log_likelihood(model: LanguageModel, some_text: str):
    #print("model is {}text is{} ".format(model,some_text))
    a = model.predict_all(some_text)
    #print(a.shape)
    #print(a)
    #b = -a
    s = utils.one_hot(some_text)
    t= a[:,:-1]*(s)
    return t.sum()
    #print(s.shape)
    #print(s)
    #print("sum is {}".format(b.sum()))
    #return b.sum()
    #print (b)
    s = utils.one_hot(some_text)
    #print(s.shape)
    #print(s)
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    raise NotImplementedError('log_likelihood')


def sample_random(model: LanguageModel, max_length: int = 100):
    
    S = ""
    for i in range(max_length):
      #data = utils.one_hot(S)
      #print(data.shape)
      o = model.predict_all(S)
      #print(o)
      #y = torch.nn.Softmax()
      print(torch.squeeze(o).shape)
      #o = y(torch.squeeze(o))
      o = torch.squeeze(o)
      hot = utils.one_hot(S)
      print("hot shape is {}".format(hot.shape))
      #print(o)
      #print ("o shape is  {}".format(o.shape))
      #print("o[-1] shape is {}".format(o[:,-1].shape))   
      if(i == 0):
          s = torch.distributions.Categorical(logits = o).sample()
      else:
          print("o's shape is {}".format(o[:,-2].shape))
          s = torch.distributions.Categorical(logits = o[:,-1]).sample()
      #print(s)
      #print("char is {}".format(char_set[s.item()]))
      S = S+(char_set[s.item()])
      if (char_set[s.item()] == '.'):
        break
      print(S)
    return (S)
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    raise NotImplementedError('sample_random')


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    
    strings =[]
    log_likel = []
    h = TopNHeap(n_results)
    for i in range(30):
      string1 = sample_random(model,max_length)
      print(string1)
      strings.append(string1)
      log_like = log_likelihood(model,string1)
      if(average_log_likelihood == True):
        log_like = log_like/len(string1)
      log_likel.append(log_like.item())
      h.add(log_like.item())
    print("h elements are {}".format(h.elements))
    print("the strings are {}".format(strings))
    print("likelihood is {}".format(log_likel))

    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    raise NotImplementedError('beam_search')


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
