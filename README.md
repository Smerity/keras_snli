# Keras SNLI example

This repository contains Keras code to train a variety of neural networks to tackle the [Stanford Natural Language Inference (SNLI) corpus](http://nlp.stanford.edu/projects/snli/).

The aim is to determine whether a premise sentence is entailed, neutral, or contradicts a hypothesis sentence - i.e. "A soccer game with multiple males playing" entails "Some men are playing a sport" while "A black race car starts up in front of a crowd of people" contradicts "A man is driving down a lonely road".

The model architecture is:

+ Extract a 300D word vector from the fixed GloVe vocabulary
+ Pass the 300D word vector through a ReLU "translation" layer
+ Encode the premise and hypothesis sentences using the same encoder (summation, GRU, LSTM, ...)
+ Concatenate the two 300D resulting sentence embeddings
+ 3 layers of 600D ReLU layers
+ 3 way softmax

![Visual image description of the model](https://rawgit.com/Smerity/keras_snli/master/snli_model.svg)

Training uses RMSProp and stops when three epochs have passed with no improvement to the validation loss.
Following [Liu et al. 2016](http://arxiv.org/abs/1605.09090), the GloVe embeddings are not updated during training.
Unlike [Liu et al. 2016](http://arxiv.org/abs/1605.09090), I don't initialize out of vocabulary embeddings randomly and instead leave them zeroed.
There are likely improvements that could be made by allowing training with a strong L2 penalty when moving away from the GloVe embeddings.

One of the most important aspects when we used fixed Glove embeddings is the "translation" layer.
[Bowman et al. 2016](http://nlp.stanford.edu/pubs/snli_paper.pdf) use 

The model is relatively simple yet sits at a far higher level than other comparable baselines (specifically summation, GRU, and LSTM models) listed on [the SNLI page](http://nlp.stanford.edu/projects/snli/).

Model                                              | Parameters | Train  | Validation | Test
---                                                | ---        | ---    | ---        | ---
300D sum(word vectors) + 3 x 600D ReLU (this code) | 1.2m       | 0.831  | 0.823      | 0.825
300D GRU + 3 x 600D ReLU (this code)               | 1.7m       | 0.843  | 0.830      | 0.823
300D LSTM + 3 x 600D ReLU (this code)              | 1.9m       | 0.855  | 0.829      | 0.823
--                                                | ---        | ---    | ---        | ---
300D LSTM encoders (Bowman et al. 2016)            | 3.0m       | 0.839  | -          | 0.806
1024D GRU w/ unsupervised 'skip-thoughts' pre-training (Vendrov et al. 2015) | 15m | 0.988 | - | 0.814
300D Tree-based CNN encoders (Mou et al. 2015)     | 3.5m       | 0.833  | -          | 0.821
300D SPINN-PI encoders (Bowman et al. 2016)        | 3.7m       | 0.892  | -          | 0.832
600D (300+300) BiLSTM encoders (Liu et al. 2016)   | 3.5m       | 0.833  | -          | 0.834

Only the numbers for pure sentential embedding models are shown here.
The SNLI homepage shows the full list of models where attentional models perform better.
If I've missed including any comparable models, submit a pull request.

All models could benefit from a more thorough evaluation and/or grid search as the existing parameters are guesstimates inspired by various papers (Bowman et al. 2015, Bowman et al. 2016, Liu et al. 2016).
That the summation of word embeddings (jokingly referred to as SumRNN) performs so well compared to GRUs or LSTMs is a surprise and warrants additional investigation.
Further work should be done exploring The hyperparameters of the GRU and LSTM such that they beat the SumRNN.
