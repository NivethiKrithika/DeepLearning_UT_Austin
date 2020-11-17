def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    
    strings =[]
    log_likel = []
    final_list = []
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
    for ele in h.elements:
      ind = log_likel.index(ele)
      final_list.append(strings[ind])
    return (final_list)
