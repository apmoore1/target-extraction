# [Attention-based LSTM for Aspect-level Sentiment Classification](https://aclweb.org/anthology/D16-1058)
This paper created the ATAE, AT-LSTM, and AE-LSTM models. The first two models are best described in the images below where as the second in this code base is an adaptation of the AE-LSTM shown in the last/third image below, where instead of just using the last hidden state in the LSTM we applied the same attention mechanism as applied to the ATAE and AT-LSTM models just we did not concat the target vector again after the words have been contextualised by the LSTM. 

![alt text](./ATAE%20figure.png "ATAE architecture")
![alt text](./AT%20figure.png "AT-LSTM architecture")
![alt text](./AE%20figure.png "Original AE-LSTM architecture")

These figures have been taken from the original paper from figures 3, 2, and 1 respectively.

The configuration files that replicate ATAE, AT-LSTM, and our attention based version of AE-LSTM can be found at [atae.jsonnet](./atae.jsonnet), [at.jsonnet](./at.jsonnet), and [ae.jsonnet](./ae.jsonnet) respectively.
