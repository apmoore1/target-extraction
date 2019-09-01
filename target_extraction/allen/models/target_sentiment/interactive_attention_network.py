from typing import Optional, List, Dict

from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules.attention import BilinearAttention
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import Activation, util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import torch
from torch.nn.modules import Dropout, Linear

from target_extraction.allen.models import target_sentiment

@Model.register("interactive_attention_network_classifier")
class InteractivateAttentionNetworkClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 target_encoder: Seq2SeqEncoder,
                 feedforward: Optional[FeedForward] = None,
                 context_attention_activation_function: str = 'tanh',
                 target_attention_activation_function: str = 'tanh',
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 dropout: float = 0.0,
                 label_name: str = 'target-sentiment-labels',
                 loss_weights: Optional[List[float]] = None) -> None:
        super().__init__(vocab, regularizer)
        '''
        :param vocab: A Vocabulary, required in order to compute sizes 
                      for input/output projections.
        :param context_field_embedder: Used to embed the context/sentence and 
                                       target text if target_field_embedder is 
                                       None but the target_encoder is NOT None.
        :param context_encoder: Encoder that will create the representation 
                                for the sentence/context that the target 
                                appears in.
        :param target_encoder: Encoder that will create the representation of 
                               target text tokens.
        :param feedforward: An optional feed forward layer to apply after the 
                            encoder.
        :param context_attention_activation_function: The attention method to be
                                                      used on the context.
        :param target_attention_activation_function: The attention method to be
                                                     used on the target text.
        :param target_field_embedder: Used to embed the target text to give as 
                                      input to the target_encoder. Thus this 
                                      allows a separate embedding for context 
                                      and target text.
        
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
        
        This is based on the `Interactive Attention Networks for Aspect-Level 
        Sentiment Classification 
        <https://www.ijcai.org/proceedings/2017/0568.pdf>`_. The model is also 
        known as `IAN`.

         .. _variational dropout:
           https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        '''

        self.label_name = label_name
        self.context_field_embedder = context_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size(self.label_name)
        self.target_encoder = target_encoder
        self.context_encoder = context_encoder
        self.feedforward = feedforward

        context_attention_activation_function = Activation.by_name(f'{context_attention_activation_function}')()
        target_attention_activation_function = Activation.by_name(f'{target_attention_activation_function}')()
        
        target_encoder_out = self.target_encoder.get_output_dim()
        context_encoder_out = self.context_encoder.get_output_dim()
        self.context_attention_layer = BilinearAttention(target_encoder_out,
                                                         context_encoder_out,
                                                         context_attention_activation_function,
                                                         normalize=True)
        self.target_attention_layer = BilinearAttention(context_encoder_out,
                                                        target_encoder_out,
                                                        target_attention_activation_function,
                                                        normalize=True)
        # To be used as the pooled input into the target attention layer as 
        # the query vector.
        self._context_averager = BagOfEmbeddingsEncoder(context_encoder_out, 
                                                        averaged=True)
        # To be used as the pooled input into the context attention layer as 
        # the query vector.
        self._target_averager = BagOfEmbeddingsEncoder(target_encoder_out, 
                                                       averaged=True)

        # Set the loss weights (have to sort them by order of label index in 
        # the vocab)
        self.loss_weights = target_sentiment.util.loss_weight_order(self, loss_weights, self.label_name)

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = target_encoder_out + context_encoder_out
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

        # Ensure that the dimensions of the text field embedder and text encoder
        # match
        check_dimensions_match(context_field_embedder.get_output_dim(), 
                               context_encoder.get_input_dim(),
                               "context field embedding dim", "text encoder input dim")
        # Ensure that the dimensions of the target or text field embedder and 
        # the target encoder match
        target_field_embedder_dim = context_field_embedder.get_output_dim()
        target_field_error = "context field embedding dim"
        if self.target_field_embedder:
            target_field_embedder_dim = target_field_embedder.get_output_dim()
            target_field_error = "target field embedding dim"
        
        check_dimensions_match(target_field_embedder_dim, 
                               target_encoder.get_input_dim(),
                               target_field_error, "target encoder input dim")
        
        # TimeDistributed anything that is related to the targets.
        self.target_encoder = TimeDistributed(self.target_encoder)
        if self.feedforward is not None:
            self.feedforward = TimeDistributed(self.feedforward)
        self.label_projection = TimeDistributed(self.label_projection)
        self._time_variational_dropout = TimeDistributed(self._variational_dropout)
        self._time_naive_dropout = TimeDistributed(self._naive_dropout)

        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                targets: Dict[str, torch.LongTensor],
                target_sentiments: torch.LongTensor = None,
                metadata: torch.LongTensor = None, 
                **kwargs
                ) -> Dict[str, torch.Tensor]:
        '''
        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        # Embed and encode text as a sequence
        embedded_context = self.context_field_embedder(tokens)
        embedded_context = self._variational_dropout(embedded_context)
        context_mask = util.get_text_field_mask(tokens)

        # Size (batch size, sequence length, embedding dim)
        encoded_context_seq = self.context_encoder(embedded_context, context_mask)
        encoded_context_seq = self._variational_dropout(encoded_context_seq)
        _, context_sequence_length, context_dim = encoded_context_seq.shape

        # Embed and encode target as a sequence
        if self.target_field_embedder:
            embedded_targets = self.target_field_embedder(targets)
        else:
            embedded_targets = self.context_field_embedder(targets)
        # Size (batch size, num targets, sequence length, embedding dim)
        embedded_targets = self._time_variational_dropout(embedded_targets)
        targets_mask = util.get_text_field_mask(targets, num_wrapping_dims=1)

        encoded_targets_seq = self.target_encoder(embedded_targets, targets_mask)
        encoded_targets_seq = self._time_variational_dropout(encoded_targets_seq)
        batch_size, number_targets, target_sequence_length, encoded_target_dim = encoded_targets_seq.shape

        #
        # Attention layers
        #
        # context attention
        # Get average of the target hidden states as the query vector for the 
        # context attention. Need to reshape the context so there are enough 
        # contexts per target so that attention can be applied
        batch_size_num_targets = batch_size * number_targets
        # Batch Size * Number targets, sequence length, dim
        repeated_context_seq = encoded_context_seq.unsqueeze(1).repeat(1, number_targets, 1, 1)
        repeated_context_seq = repeated_context_seq.view(batch_size_num_targets, context_sequence_length, context_dim)
        # Batch size * Number targets, sequence length
        repeated_context_mask = context_mask.unsqueeze(1).repeat(1, number_targets, 1)
        repeated_context_mask = repeated_context_mask.view(batch_size_num_targets, context_sequence_length)
        # Need to reshape the enocded target sequences so that they are 
        # Batch size * number targets, target length, dim
        seq_encoded_targets_seq = encoded_targets_seq.view(batch_size_num_targets, 
                                                           target_sequence_length,
                                                           encoded_target_dim)
        seq_targets_mask = targets_mask.view(batch_size_num_targets, target_sequence_length)
        # Batch size * number targets, number targets, dim
        avg_targets_vec = self._target_averager(seq_encoded_targets_seq, seq_targets_mask)
        # Batch size * number targets, sequence length
        context_attentions_weights = self.context_attention_layer(avg_targets_vec,
                                                                  repeated_context_seq,
                                                                  repeated_context_mask)
        # Convert into Batch Size, Number Targets, sequence length
        context_attentions_weights = context_attentions_weights.view(batch_size, number_targets, context_sequence_length)
        # Convert into Batch Size, Number Targets, sequence length, dim
        weighted_encoded_context_seq = repeated_context_seq.view(batch_size, number_targets, context_sequence_length, context_dim)
        # Batch size, number targets, sequence length, dim
        context_attentions_weights = context_attentions_weights.unsqueeze(-1)
        weighted_encoded_context_seq = weighted_encoded_context_seq * context_attentions_weights
        # Batch size, number targets, dim
        weighted_encoded_context_vec = weighted_encoded_context_seq.sum(2)
        
        # target attention
        # Get average of the context hidden states as the query vector for the 
        # target attention. Need to do the same as for the context, reshape 
        # the average context so that there are the same number of contexts 
        # vectors as targets for that context
        # Batch size, dim
        avg_context_vec = self._context_averager(encoded_context_seq, context_mask)
        repeated_avg_context_vec = avg_context_vec.unsqueeze(1).repeat(1,number_targets,1)
        repeated_avg_context_vec = repeated_avg_context_vec.view(batch_size_num_targets, context_dim)
        # Need to reshape the enocded target sequences so that they are 
        # Batch size * number targets, target length, dim
        seq_encoded_targets_seq = encoded_targets_seq.view(batch_size_num_targets, 
                                                           target_sequence_length,
                                                           encoded_target_dim)
        seq_targets_mask = targets_mask.view(batch_size_num_targets, target_sequence_length)
        # batch size * number targets, target length
        target_attention_weights = self.target_attention_layer(repeated_avg_context_vec,
                                                               seq_encoded_targets_seq,
                                                               seq_targets_mask)
        # batch size, number targets, target length
        target_attention_weights = target_attention_weights.view(batch_size, number_targets, target_sequence_length)
        target_attention_weights = target_attention_weights.unsqueeze(-1)
        weighted_encoded_target_seq = encoded_targets_seq * target_attention_weights
        # batch size, number targets, dim
        weighted_encoded_target_vec = weighted_encoded_target_seq.sum(2)
        
        # Concatenate the two weighted context and target vectors
        weighted_text_target = torch.cat([weighted_encoded_context_vec, 
                                          weighted_encoded_target_vec], -1)
        weighted_text_target = self._time_naive_dropout(weighted_text_target)

        if self.feedforward:
            weighted_text_target = self.feedforward(weighted_text_target)
        # Putting it through a tanh first and then a softmax
        logits = self.label_projection(weighted_text_target)
        logits = torch.tanh(logits)
        label_mask = util.get_text_field_mask(targets)
        masked_class_probabilities = util.masked_softmax(logits, label_mask.unsqueeze(-1))

        output_dict = {"class_probabilities": masked_class_probabilities, 
                       "targets_mask": label_mask}

        if target_sentiments is not None:
            # gets the loss per target instance due to the average=`token`
            if self.loss_weights is not None:
                loss = util.sequence_cross_entropy_with_logits(logits, target_sentiments, 
                                                               label_mask, average='token',
                                                               alpha=self.loss_weights)
            else:
                loss = util.sequence_cross_entropy_with_logits(logits, target_sentiments, 
                                                               label_mask, average='token')
            for metrics in [self.metrics, self.f1_metrics]:
                for metric in metrics.values():
                    metric(logits, target_sentiments, label_mask)
            output_dict["loss"] = loss

        if metadata is not None:
            words = []
            texts = []
            targets = []
            target_words = []
            # Attention weights and the related masks
            target_attention = []
            targets_word_mask = []
            word_attention = []
            word_mask = []
            for batch_index, sample in enumerate(metadata):
                words.append(sample['text words'])
                texts.append(sample['text'])
                word_attention.append(context_attentions_weights[batch_index])
                word_mask.append(context_mask[batch_index])
                targets.append(sample['targets'])
                target_words.append(sample['target words'])
                target_attention.append(target_attention_weights[batch_index])
                targets_word_mask.append(targets_mask[batch_index])
            output_dict["words"] = words
            output_dict["text"] = texts
            output_dict["word_attention"] = word_attention
            output_dict["word_mask"] = word_mask
            output_dict["targets"] = targets
            output_dict["target words"] = target_words
            output_dict["targets_attention"] = target_attention
            output_dict["targets_word_mask"] = targets_word_mask

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

        