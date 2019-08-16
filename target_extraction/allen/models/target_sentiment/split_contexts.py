from typing import Dict, Optional

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout, Linear

@Model.register("split_contexts_classifier")
class SplitContextsClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 left_text_encoder: Seq2VecEncoder,
                 right_text_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 target_encoder: Optional[Seq2VecEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 word_dropout: float = 0.0,
                 dropout: float = 0.0,
                 label_name: str = 'target-sentiment-labels') -> None:
        super().__init__(vocab, regularizer)
        '''
        :param vocab: A Vocabulary, required in order to compute sizes 
                      for input/output projections.
        :param text_field_embedder: Used to embed the text and target text if
                                    target_field_embedder is None but the 
                                    target_encoder is NOT None.
        :param left_text_encoder: Encoder that will create the representation 
                                  of the tokens left of the target and  
                                  the target itself if included from the 
                                  dataset reader.
        :param right_text_encoder: Encoder that will create the representation 
                                   of the tokens right of the target and the 
                                   target itself if included from the 
                                   dataset reader.
        :param feedforward: An optional feed forward layer to apply after the 
                            encoder.
        :param target_field_embedder: Used to embed the target text to give as 
                                      input to the target_encoder. Thus this 
                                      allows a seperate embedding for text and 
                                      target text.
        :param target_encoder: Encoder that will create the representation of 
                               target text tokens.
        :param initializer: Used to initialize the model parameters.
        :param regularizer: If provided, will be used to calculate the 
                            regularization penalty during training.
        :param dropout: To apply dropout after each layer apart from the last 
                        layer. All dropout that is applied to timebased data 
                        will be `variational dropout`_ all else will be  
                        standard dropout.
        :param label_name: Name of the label name space.
        
        Without the target encoder this will be the standard TDLSTM method 
        from `Effective LSTM's for Target-Dependent Sentiment classification`_
        . With the target encoder this will then become the TCLSTM method 
        from `Effective LSTM's for Target-Dependent Sentiment classification`_.
        .. _variational dropout:
           https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        .. _Effective LSTM's for Target-Dependent Sentiment classification:
           https://aclanthology.coli.uni-saarland.de/papers/C16-1311/c16-1311
        '''

        self.label_name = label_name
        self.text_field_embedder = text_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size(self.label_name)
        self.left_text_encoder = left_text_encoder
        self.right_text_encoder = right_text_encoder
        self.target_encoder = target_encoder
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            left_out_dim = self.left_text_encoder.get_output_dim()
            right_out_dim = self.right_text_encoder.get_output_dim()
            output_dim = left_out_dim + right_out_dim
        self.label_projection = Linear(output_dim, self.num_classes)
        
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.f1_metrics = {}
        # F1 Scores
        label_index_name = self.vocab.get_index_to_token_vocabulary(self.label_name)
        for label_index, _label_name in label_index_name.items():
            _label_name = f'F1_{_label_name.capitalize()}'
            self.f1_metrics[_label_name] = F1Measure(label_index)
        # Dropout
        self._variational_dropout = InputVariationalDropout(dropout)
        self._naive_dropout = Dropout(dropout)
        
        self.loss = torch.nn.CrossEntropyLoss()

        # Ensure that the input to the right_text_encoder and left_text_encoder
        # is the size of the target encoder output plus the size of the text 
        # embedding output.
        if self.target_encoder:
            right_in_dim = self.right_text_encoder.get_input_dim()
            left_in_dim = self.left_text_encoder.get_input_dim()

            target_dim = self.target_encoder.get_output_dim()
            text_dim = self.text_field_embedder.get_output_dim()
            total_out_dim = target_dim + text_dim
            config_err_msg = ("As the target is being encoded the output of the" 
                              " target encoder is concatenated onto each word "
                              " vector for the left and right contexts " 
                              "therefore the input of the right_text_encoder"
                              "/left_text_encoder is the output dimension of "
                              "the target encoder + the dimension of the word "
                              "embeddings for the left and right contexts.")
            
            if (total_out_dim != right_in_dim or 
                total_out_dim != left_in_dim):
                raise ConfigurationError(config_err_msg)
        # Ensure that the target field embedder has an output dimension the 
        # same as the input dimension to the target encoder.
        if self.target_encoder and self.target_field_embedder:
            target_embed_out = self.target_field_embedder.get_output_dim()
            target_in = self.target_encoder.get_input_dim()
            config_embed_err_msg = ("The Target field embedder should have"
                                    " the same output size "
                                    f"{target_embed_out} as the input to "
                                    f"the target encoder {target_in}")
            if target_embed_out != target_in:
                raise ConfigurationError(config_embed_err_msg)

        initializer(self)

    def forward(self, left_contexts: Dict[str, torch.LongTensor],
                right_contexts: Dict[str, torch.LongTensor],
                targets: Dict[str, torch.LongTensor],
                target_sentiments: torch.LongTensor = None,
                metadata: torch.LongTensor = None, **kwargs
                ) -> Dict[str, torch.Tensor]:
        '''
        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        # Batch size, number of targets, sequence length, vector dimension
        left_embedded_text = self.text_field_embedder(left_contexts)
        left_embedded_text = self._variational_dropout(left_embedded_text)
        left_text_mask = util.get_text_field_mask(left_contexts, num_wrapping_dims=1)

        right_embedded_text = self.text_field_embedder(right_contexts)
        right_embedded_text = self._variational_dropout(right_embedded_text)
        right_text_mask = util.get_text_field_mask(right_contexts, num_wrapping_dims=1)
        print('here')
        if self.target_encoder:
            if self.target_field_embedder:
                embedded_target = self.target_field_embedder(targets)
            else:
                embedded_target = self.text_field_embedder(targets)
            embedded_target = self._variational_dropout(embedded_target)
            target_text_mask = util.get_text_field_mask(targets, num_wrapping_dims=1))

            target_encoded_text = self.target_encoder(embedded_target, 
                                                      target_text_mask)
            target_encoded_text = self._naive_dropout(target_encoded_text)
            # Encoded target to be of dimension (batch, Number of Targets, words, dim) 
            # currently (batch, Number of Targets, dim)
            target_encoded_text = target_encoded_text.unsqueeze(1)

            # Need to repeat the target word for each word in the left 
            # and right word.
            left_num_padded = left_embedded_text.shape[1]
            right_num_padded = right_embedded_text.shape[1]

            left_targets = target_encoded_text.repeat((1, 1, left_num_padded, 1))
            right_targets = target_encoded_text.repeat((1, 1, right_num_padded, 1))
            # Add the target to each word in the left and right contexts
            left_embedded_text = torch.cat((left_embedded_text, left_targets), -1)
            right_embedded_text = torch.cat((right_embedded_text, right_targets), -1)
        
        
        left_encoded_text = self.left_text_encoder(left_embedded_text, 
                                                   left_text_mask)
        left_encoded_text = self._naive_dropout(left_encoded_text)

        right_encoded_text = self.right_text_encoder(right_embedded_text, 
                                                     right_text_mask)
        right_encoded_text = self._naive_dropout(right_encoded_text)


        encoded_left_right = torch.cat([left_encoded_text, right_encoded_text], 
                                       dim=-1)
        if self.feedforward:
            encoded_left_right = self.feedforward(encoded_left_right)
        logits = self.label_projection(encoded_left_right)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"class_probabilities": class_probabilities}

        if target_sentiments is not None:
            loss = self.loss(logits, target_sentiments)
            for metrics in [self.metrics, self.f1_metrics]:
                for metric in metrics.values():
                    metric(logits, target_sentiments)
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        '''
        Adds the predicted label to the output dict.
        '''
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace=self.label_name)
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict    

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Other scores
        metric_name_value = {}
        for metric_name, metric in self.metrics.items():
            metric_name_value[metric_name] = metric.get_metric(reset)
        # F1 scores
        all_f1_scores = []
        for metric_name, metric in self.f1_metrics.items():
            precision, recall, f1_measure = metric.get_metric(reset)
            all_f1_scores.append(f1_measure)
            metric_name_value[metric_name] = f1_measure
        metric_name_value['Macro_F1'] = sum(all_f1_scores) / len(self.f1_metrics)
        return metric_name_value