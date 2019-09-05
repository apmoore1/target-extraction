from typing import Dict, Optional, List

from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy
import torch
from torch.nn.modules import Dropout, Linear

from target_extraction.allen.models import target_sentiment
from target_extraction.allen.models.target_sentiment.util import elmo_input_reshape, elmo_input_reverse
    
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
                 dropout: float = 0.0,
                 label_name: str = 'target-sentiment-labels',
                 loss_weights: Optional[List[float]] = None) -> None:
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
        :param loss_weights: The amount of weight to give the negative, neutral,
                             positive classes respectively. e.g. [0.2, 0.5, 0.3]
                             would weight the negative class by a factor of 
                             0.2, neutral by 0.5 and positive by 0.3. NOTE It 
                             assumes the sentiment labels are the following:
                             [negative, neutral, positive].
        
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
        
        # Set the loss weights (have to sort them by order of label index in 
        # the vocab)
        self.loss_weights = target_sentiment.util.loss_weight_order(self, loss_weights, self.label_name)

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
        

        # Ensure that the input to the right_text_encoder and left_text_encoder
        # is the size of the target encoder output plus the size of the text 
        # embedding output.
        if self.target_encoder is not None:
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
            check_dimensions_match(target_in, target_embed_out, 
                                   'target_field_embedder output', 
                                   'target_encoder input')

        # TimeDistributed everything as we are processing multiple Targets at 
        # once as the input is a sentence containing one or more targets
        self.left_text_encoder = TimeDistributed(self.left_text_encoder)
        self.right_text_encoder = TimeDistributed(self.right_text_encoder)
        if self.target_encoder is not None:
            self.target_encoder = TimeDistributed(self.target_encoder)
        if self.feedforward is not None:
            self.feedforward = TimeDistributed(self.feedforward)
        self.label_projection = TimeDistributed(self.label_projection)
        self._variational_dropout = TimeDistributed(self._variational_dropout)
        self._naive_dropout = TimeDistributed(self._naive_dropout)

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
        targets_mask = util.get_text_field_mask(targets)
        batch_size, number_targets = targets_mask.shape
        batch_size_num_targets = batch_size * number_targets

        temp_left_contexts = elmo_input_reshape(left_contexts, batch_size, 
                                                number_targets, batch_size_num_targets)
        left_embedded_text = self.text_field_embedder(temp_left_contexts)
        left_embedded_text = elmo_input_reverse(left_embedded_text, left_contexts, 
                                                batch_size, number_targets, 
                                                batch_size_num_targets)
        left_embedded_text = self._variational_dropout(left_embedded_text)
        left_text_mask = util.get_text_field_mask(left_contexts, num_wrapping_dims=1)

        temp_right_contexts = elmo_input_reshape(right_contexts, batch_size, 
                                                 number_targets, batch_size_num_targets)
        right_embedded_text = self.text_field_embedder(temp_right_contexts)
        right_embedded_text = elmo_input_reverse(right_embedded_text, right_contexts, 
                                                 batch_size, number_targets, 
                                                 batch_size_num_targets)
        right_embedded_text = self._variational_dropout(right_embedded_text)
        right_text_mask = util.get_text_field_mask(right_contexts, num_wrapping_dims=1)
        if self.target_encoder:
            temp_target = elmo_input_reshape(targets, batch_size, number_targets, 
                                             batch_size_num_targets)
            if self.target_field_embedder:
                embedded_target = self.target_field_embedder(temp_target)
            else:
                embedded_target = self.text_field_embedder(temp_target)
            embedded_target = elmo_input_reverse(embedded_target, targets, 
                                                 batch_size, number_targets, 
                                                 batch_size_num_targets)
            embedded_target = self._variational_dropout(embedded_target)
            target_text_mask = util.get_text_field_mask(targets, num_wrapping_dims=1)

            target_encoded_text = self.target_encoder(embedded_target, 
                                                      target_text_mask)
            target_encoded_text = self._naive_dropout(target_encoded_text)
            # Encoded target to be of dimension (batch, Number of Targets, words, dim) 
            # currently (batch, Number of Targets, dim)
            target_encoded_text = target_encoded_text.unsqueeze(2)

            # Need to repeat the target word for each word in the left 
            # and right word.
            left_num_padded = left_embedded_text.shape[2]
            right_num_padded = right_embedded_text.shape[2]

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

        masked_class_probabilities = util.masked_softmax(logits, targets_mask.unsqueeze(-1))

        output_dict = {"class_probabilities": masked_class_probabilities, 
                       "targets_mask": targets_mask}

        if target_sentiments is not None:
            # gets the loss per target instance due to the average=`token`
            if self.loss_weights is not None:
                loss = util.sequence_cross_entropy_with_logits(logits, target_sentiments, 
                                                               targets_mask, average='token',
                                                               alpha=self.loss_weights)
            else:
                loss = util.sequence_cross_entropy_with_logits(logits, target_sentiments, 
                                                               targets_mask, average='token')
            for metrics in [self.metrics, self.f1_metrics]:
                for metric in metrics.values():
                    metric(logits, target_sentiments, targets_mask)
            output_dict["loss"] = loss

        if metadata is not None:
            words = []
            texts = []
            targets = []
            target_words = []
            for sample in metadata:
                words.append(sample['text words'])
                texts.append(sample['text'])
                targets.append(sample['targets'])
                target_words.append(sample['target words'])
            output_dict["words"] = words
            output_dict["text"] = texts
            output_dict["targets"] = targets
            output_dict["target words"] = target_words

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        '''
        Adds the predicted label to the output dict, also removes any class 
        probabilities that do not have a target associated which is caused 
        through the batch prediction process and can be removed by using the 
        target mask.
        '''
        batch_target_predictions = output_dict['class_probabilities'].cpu().data.numpy()
        target_masks = output_dict['targets_mask'].cpu().data.numpy()
        # Should have the same batch size and max target nubers
        batch_size = batch_target_predictions.shape[0]
        max_number_targets = batch_target_predictions.shape[1]
        assert target_masks.shape[0] == batch_size
        assert target_masks.shape[1] == max_number_targets

        sentiments = []
        non_masked_class_probabilities = []
        for batch_index in range(batch_size):
            target_sentiments = []
            target_non_masked_class_probabilities = []

            target_predictions = batch_target_predictions[batch_index]
            target_mask = target_masks[batch_index]
            for index, target_prediction in enumerate(target_predictions):
                if target_mask[index] != 1:
                    continue
                label_index = numpy.argmax(target_prediction)
                label = self.vocab.get_token_from_index(label_index, 
                                                        namespace=self.label_name)
                target_sentiments.append(label)
                target_non_masked_class_probabilities.append(target_prediction)
            sentiments.append(target_sentiments)
            non_masked_class_probabilities.append(target_non_masked_class_probabilities)
        output_dict['sentiments'] = sentiments
        output_dict['class_probabilities'] = non_masked_class_probabilities
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