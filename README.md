## Description

This repository presents a general framework for using RNNs in modelling language transduction tasks. Customized pipelines for training and deploying your own models given properly formatted data and a corresponding `config.json` file are also provided. Tutorials for building your own pipelines or understanding the code may come later. I may also create another similar repository for RNN seq2seq models. 

This repository is a result of my project [rnn-seq2seq-learning](https://github.com/jaaack-wang/rnn-seq2seq-learning).



## Usage

**Requirements**: PyTorch (I used version 1.10.1), Matplotlib (I used version 3.4.2)

To use the customized pipelines for training and deploying your own models, you need to have training data that consists of input-output sequence pairs separated by a tab (i.e., `\t`). Moreover, both the input and output sequences should be presented in a way such that either each symbol in the sequence is a token already or the tokens can be easily inferred from a separator that is not part of the token list. If the latter is the case, you need to specify the separator(s) used in your `config.json` file, which is responsible for configuring the training and inference processes.

The `config.json` file can be created by using the templates provided in `example_configs`. The `config.json` file contains three types of configurations, one for the RNN model (i.e., `ModelConfig`), one for the training (i.e., `TrainConfig`), and one for the inference (i.e., `InferConfig`). If you use a separator in your data file, this information should be incorporated in the `TrainConfig` for training and `InferConfig` for depolyment. There are three types of RNN models available, i.e., SRNN (Simple RNN), GRU (Gated Recurrent Unit), and LSTM (Long Short-Term Memory).

To run the training pipeline, use the following command line. The outputs of the training are: a best model saved during training, training logs (including an optional plot), and the used configurations that will be re-used for inference. 

```cmd
python3 scripts/train.py --config_fp filepath_to_the_config_file
```

To run the inference pipeline, make sure you have a text file that consists of only the input sequences that you want to transduce (one example per line) and use the following command line:

```cmd
python3 scripts/predict.py --config_fp filepath_to_the_config_file
```

For example, you can do the following to test both pipelines on the toy examples provided by this repo:

```cmd
python3 scripts/train.py --config_fp example_configs/identity_config.json
python3 scripts/predict.py --config_fp example_configs/identity_config.json
```

Both pipelines are highly customized as a tradeoff for simplicity. 

## What RNNs can transduce

It turns out that RNNs can model transduction tasks very robustly where each input symbol corresponds to only one determined output symbol. In this case, only a few training examples that cover all the possible transduction are needed. For example, to learn identity function where each input symbol is mapped to itself, this can be made possible by one example where the sequence is a list of all possible tokens, and a hidden state size as small as 2, although more examples greatly accelerate learning speed. I have trained such models and extensively tested them on sequences of over 10,000 symbols without an error. 

In the provided examples (check `data`, `example_configs`, and `outputs`), I used RNNs to model the following functions (where the vocabulary is simply the 26 lowercase English letters): 

- identity function: map each input symbol to itself
- string to number function: map each input symbol to a number (0-25).
- vowel deletion function: map each input symbol to itself except for vowels (i.e., a, e, i, o, u), which are mapped to a space, denoting `deletion`
- Initial CV reduplication function: if a sequence begins with a consonant (anything but a vowel) followed by a vowel, then reduplicate these two symbols; otherwise, map each input symbol to itself (e.g., *b-a-->b-aba*, *b-b-a-->b-b-a*). The hyphen denotes how the input and output sequences should be tokenized in the training data.   
- Reduplicated initial CV reversion function: undo what Initial CV reduplication function does (e.g., *b-aba-->b-a*, *b-b-a-->b-b-a*)..

It turns out that the **initial CV reduplication function** is the hardest to learn, which requires more training examples, and a larger sized model (but still very small), because some outputs are conditioned on the construction of the input sequences. I tried several RNN models on 701 pseudo examples, covering all the possible consonant initials plus some random examples, but the trained models still cannot achieve 100% accuracy on unseen examples. However, a well-trained model can robustly achieve over 90% accuracy even for sequences much longer than those seen during training. 

Please note that, **reduplicated initial CV reversion** is way much easier because each input symbol is mapped to an deterministic output symbol, so it can be learnt perfectly (or nearly perfectly). 



## Limitations

Nevertheless, RNNs **are not good for practical use** because it can only produce an output sequence of the **same length** with the input sequence, **in terms of the number of tokens**. Even for the initial CV reduplication function where the output sequence  seems to be longer than the input sequence, the trick is, the RNNs take those *vcv* structures as speicifc tokens for the output sequences, rather than three individual tokens as in the case of the input sequences. Similarly, for the reduplicated initial CV reversion function, it is the opposite. 

In other words, to make RNNs useful for transduction tasks in real world, we need to first have well aligned input-output sequence pairs to train the models, which, however, is usually the hardest thing to do. Because if we know in advance how each input should be mapped to the output, we can simply use symbolic approaches to do the modelling. 

Moreover, RNNs are known to be comparable to a one-way finite state transducer, and as a result, it will never be able to learn a transduction function that requires a 2-way finite state transducer. For example, RNNs cannot learn how to reverse an input sequence simply because there is no way for it to know what comes in the end beforehand. Even bidirectional RNNs cannot do that because the information retrieved from backward hidden states vanishes as the input sequnece grows in length.



## Notes

RNN seq2seq models are more flexible in modelling input-output transduction from end to end without needing aligned parallel examples, because the encoder and decoder are both RNNs that can process and produce variable-length sequences, respectively. However, for RNN seq2seq models, the problem lies in how to utilize the information encoded by the encoder during the decoding phase. For example, learning identity function is a hard problem for a pure RNN seq2seq model for at least two reasons. First, the information passed to the decoder contains vanishing information about the initial part of the input sequence, so it becomes impossible for the decoder to recall the initial input symbols as the input sequence becomes longer. Second, unlike RNNs, which produce an output symbol given an input symbol **by design**, the decoder of a pure RNN seq2seq model needs to figure out when to stop **by learning**. In this sense, pure RNN seq2seq models are not necessarily more powerful than RNN models.

In real world, however, nobody uses pure RNN seq2seq models to model anything. Instead, attention is used, which allows the decoder to access past information directly from the encoder. However, RNN seq2seq models with such a mechanism are more data-hungry and yet still have problem reliably generalizing to things that are dissimilar to the training data. For instance, their ability to capture identity function is arguably no better than the RNN models.

**The point here is that RNN models and RNN seq2seq models are different in their architectural designs, which come with different inductive biases, so they are suitable for learning different classes of transduction tasks**. It is not wise to use RNN seq2seq models to learn things we know RNN models are better at, although the former are much more convenient and flexible to use and usually yield better empirical results (with attention). Likewise, when RNN models are inpractical to use, go for other options.
