# [Effective LSTMs for Target-Dependent Sentiment Classification](https://www.aclweb.org/anthology/C16-1311)
This paper created two different models:
1. TDLSTM
2. TCLSTM

These two models are best described from the Image below (taken from figure two of the original paper). The TCLSTM architecture is the one shown in full in the figure where the target word word is concatenated onto every word in the sentence, in the TDLSTM model this is not the case which is the only difference between the two models.

As can be seen from the image the sentence is split into two contexts:
1. Left context -- starts from the left most word to the right most target word.
2. Right context -- starts from left most target word to the end of the sentence.

The left context is represented using a forward running LSTM starting from the beginning of the sentence. The right context through a backward running LSTM starting from the end of the sentence.
![alt text](./TDLSTM\ and\ TCLSTM\ figure.png "TDLSTM and TCLSTM architecture")

In this sub-folder we have 4 different training configuration files:
1. [tdlstm](./tdlstm.jsonnet) - The original TDLSTM as described in the paper.
2. [tclstm](./tclstm.jsonnet) - The original TCLSTM as described in the paper.
3. [tdlstm_excluding_target](tdlstm_excluding_target.jsonnet) - The TDLSTM model but the left and right contexts do not include the target word.
4. [tclstm_excluding_target](tclstm_excluding_target.jsonnet) - The TCLSTM model but the left and right contexts do not include the target word, but still concatenates the target word vector onto each word vector in the sentence.