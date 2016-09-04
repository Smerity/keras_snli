# Keras SNLI example

This repository contains Keras code to train a variety of neural networks to tackle the [Stanford Natural Language Inference (SNLI) corpus](http://nlp.stanford.edu/projects/snli/).

The aim is to determine whether a premise sentence is entailed, neutral, or contradicts a hypothesis sentence - i.e. "A soccer game with multiple males playing" entails "Some men are playing a sport".

+ Extract a 300D word vector from the fixed GloVe vocabulary
+ Pass the 300D word vector through a ReLU "translation" layer
+ Encode the premise and hypothesis sentences using the same encoder (summation, GRU, LSTM, ...)
+ Concatenate the two 300D resulting sentence embeddings
+ 3 layers of 600D ReLU layers
+ 3 way softmax

![Visual image description of the model](https://rawgit.com/Smerity/keras_snli/master/snli_model.svg)

Following [Liu et al. 2016](http://arxiv.org/abs/1605.09090), the GloVe embeddings are not updated during training.
Unlike [Liu et al. 2016](http://arxiv.org/abs/1605.09090), I don't initialize out of vocabulary embeddings randomly and instead leave them zeroed.

The model is relatively simple yet sits at a far higher level than other comparable baselines (specifically summation, GRU, and LSTM models) listed on [the SNLI page](http://nlp.stanford.edu/projects/snli/).

Model                                        | Parameters | Train  | Validation | Test
---                                          | ---        | ---    | ---        | ---
300D sum(words)  + Translate + 3 x 600D ReLU | 1.2m       | 0.8315 | 0.8235     | 0.8249
300D GRU + Translate + 3 x 600D ReLU         | 1.7m       | 0.8431 | 0.8303     | 0.8233
300D LSTM + Translate + 3 x 600D ReLU        | 1.9m       | 0.8551 | 0.8286     | 0.8229

All models could benefit from a more thorough evaluation and/or grid search as the existing parameters are guesstimates inspired by various papers (Bowman et al. 2015, Bowman et al. 2016, Liu et al. 2016).
That the summation of word embeddings (jokingly referred to as SumRNN) performs so well when compared to GRU or LSTM is a surprise and warrants additional investigation.
