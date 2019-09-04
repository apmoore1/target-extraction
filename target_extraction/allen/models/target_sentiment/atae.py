from typing import Optional, List, Dict

from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules import FeedForward, TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import Activation, util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Dropout, Linear
import numpy

from target_extraction.allen.models import target_sentiment

@Model.register("atae_classifier")
class ATAEClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 target_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 context_attention_activation_function: str = 'tanh',
                 target_field_embedder: Optional[TextFieldEmbedder] = None,
                 AE: bool = True, AttentionAE: bool = True,
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
        :param context_attention_activation_function: The activation function 
                                                      to be used after the 
                                                      projection of the encoded
                                                      context. (Equation 7)
                                                      in the original paper.
        :param target_field_embedder: Used to embed the target text to give as 
                                      input to the target_encoder. Thus this 
                                      allows a separate embedding for context 
                                      and target text.
        :param AE: Whether to concatentate the target representations to each 
                   words word embedding.
        :param AttentionAE: Whether to concatenate the target representations 
                            to each contextualised word representation i.e. 
                            to each word's vector after the `context_encoder` 
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
        
        This is based around the models in `Attention-based LSTM for Aspect-level 
        Sentiment Classification <https://aclweb.org/anthology/D16-1058>`_. 
        The models re-created are:
        
        1. AE-LSTM where instead of just encoding using an LSTM also applies 
           an attention network after the LSTM as in the model within 
           `Modeling Inter-Aspect Dependencies for Aspect-Based Sentiment Analysis 
           <https://www.aclweb.org/anthology/N18-2043>`_
        3. AT-LSTM
        2. ATAE

        For the 1'st model ensure `AE` is True and `AttentionAE` is False. For
        the 2'nd ensure that `AE` is False and `AttentionAE` is True. For the 
        the 3'rd ensure both `AE` and `AttentionAE` are True.

         .. _variational dropout:
           https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        '''
        if not AE and not AttentionAE:
            raise ConfigurationError('Either `AE` or `AttentionAE` have to '
                                     'be True')

        self.label_name = label_name
        self.context_field_embedder = context_field_embedder
        self.target_field_embedder = target_field_embedder
        self.num_classes = self.vocab.get_vocab_size(self.label_name)
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.feedforward = feedforward
        
        target_encoder_out = self.target_encoder.get_output_dim()
        context_encoder_out = self.context_encoder.get_output_dim()
        self.context_encoder_bidirectional = self.context_encoder.is_bidirectional()

        # Applied after the contextulisation layer and before the attention layer
        attention_projection_layer_dim = context_encoder_out
        if AttentionAE:
            attention_projection_layer_dim = context_encoder_out + target_encoder_out
        self.attention_project_layer = Linear(attention_projection_layer_dim, 
                                              attention_projection_layer_dim, 
                                              bias=False)
        self.attention_project_layer = TimeDistributed(self.attention_project_layer)

        # Activation function to be applied after projection and before attention
        context_attention_activation_function = Activation.by_name(f'{context_attention_activation_function}')()
        self._context_attention_activation_function = context_attention_activation_function
        attention_vector_dim = context_encoder_out
        if AttentionAE:
            attention_vector_dim = context_encoder_out + target_encoder_out
        self.attention_vector = Parameter(torch.Tensor(attention_vector_dim))
        self.context_attention_layer = DotProductAttention(normalize=True)

        # Final projection layers, these are applied after the attention layer
        self.final_attention_projection_layer = Linear(context_encoder_out, 
                                                       context_encoder_out, bias=False)
        self.final_hidden_state_projection_layer = Linear(context_encoder_out, 
                                                          context_encoder_out, bias=False)

        # Set the loss weights (have to sort them by order of label index in 
        # the vocab)
        self.loss_weights = target_sentiment.util.loss_weight_order(self, loss_weights, self.label_name)

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = context_encoder_out
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
        # If AE is True ensure that the context encoder input is equal to the 
        # the output of the target encoder plus the context field embedder
        context_field_embedder_out = context_field_embedder.get_output_dim()
        if AE:
            check_dimensions_match(context_field_embedder_out + target_encoder_out, 
                                   context_encoder.get_input_dim(),
                                   "context field embedding dim + Target Encoder out", 
                                   "text encoder input dim")
        else:
            check_dimensions_match(context_field_embedder_out, 
                                   context_encoder.get_input_dim(),
                                   "context field embedding dim", "text encoder input dim")
        if self.feedforward is not None:
            check_dimensions_match(context_encoder_out,
                                   self.feedforward.get_input_dim(),
                                   'Context encoder output', 
                                   'FeedForward input dim')

        self._time_variational_dropout = TimeDistributed(self._variational_dropout)

        self._AE = AE
        self._AttentionAE = AttentionAE

        self.reset_parameters()
        initializer(self)

    def reset_parameters(self):
        '''
        Intitalises the attnention vector
        '''
        torch.nn.init.uniform_(self.attention_vector, -0.01, 0.01)

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
        # Embed and encode target as a sequence
        if self.target_field_embedder:
            embedded_targets = self.target_field_embedder(targets)
        else:
            embedded_targets = self.context_field_embedder(targets)
        # Size (batch size, num targets, target sequence length, embedding dim)
        embedded_targets = self._time_variational_dropout(embedded_targets)
        targets_mask = util.get_text_field_mask(targets, num_wrapping_dims=1)

        batch_size, number_targets, target_sequence_length, target_embed_dim = embedded_targets.shape
        batch_size_num_targets = batch_size * number_targets
        encoded_targets_mask = targets_mask.view(batch_size_num_targets, target_sequence_length)
        reshaped_embedding_targets = embedded_targets.view(batch_size_num_targets, 
                                                           target_sequence_length, 
                                                           target_embed_dim)
        # Shape (Batch Size * Number targets), encoded dim
        encoded_targets_seq = self.target_encoder(reshaped_embedding_targets, encoded_targets_mask)
        encoded_targets_seq = self._naive_dropout(encoded_targets_seq)

        # Embed and encode text as a sequence
        embedded_context = self.context_field_embedder(tokens)
        embedded_context = self._variational_dropout(embedded_context)
        context_mask = util.get_text_field_mask(tokens)
        # Need to repeat the so it is of shape:
        # (Batch Size * Number Targets, Sequence Length, Dim) Currently:
        # (Batch Size, Sequence Length, Dim)
        batch_size, context_sequence_length, context_embed_dim  = embedded_context.shape
        reshaped_embedding_context = embedded_context.unsqueeze(1).repeat(1,number_targets,1,1)
        reshaped_embedding_context = reshaped_embedding_context.view(batch_size_num_targets, 
                                                                     context_sequence_length, 
                                                                     context_embed_dim)
        repeated_context_mask = context_mask.unsqueeze(1).repeat(1,number_targets,1)
        repeated_context_mask = repeated_context_mask.view(batch_size_num_targets,
                                                           context_sequence_length)
        # Need to concat the target embeddings to the context words
        repeated_encoded_targets = encoded_targets_seq.unsqueeze(1).repeat(1,context_sequence_length,1)
        if self._AE:
            reshaped_embedding_context = torch.cat((reshaped_embedding_context,repeated_encoded_targets), -1)
        # Size (batch size * number targets, sequence length, embedding dim)
        reshaped_encoded_context_seq = self.context_encoder(reshaped_embedding_context, repeated_context_mask)
        reshaped_encoded_context_seq = self._variational_dropout(reshaped_encoded_context_seq)
        # Whether to concat the aspect embeddings on to the contextualised word 
        # representations
        attention_encoded_context_seq = reshaped_encoded_context_seq
        if self._AttentionAE:
            attention_encoded_context_seq = torch.cat((attention_encoded_context_seq, repeated_encoded_targets), -1)
        _, _, attention_encoded_dim = attention_encoded_context_seq.shape

        # Projection layer before the attention layer
        attention_encoded_context_seq = self.attention_project_layer(attention_encoded_context_seq)
        attention_encoded_context_seq = self._context_attention_activation_function(attention_encoded_context_seq)
        attention_encoded_context_seq = self._variational_dropout(attention_encoded_context_seq)
        
        # Attention over the context sequence
        attention_vector = self.attention_vector.unsqueeze(0).repeat(batch_size_num_targets, 1)
        attention_weights = self.context_attention_layer(attention_vector, 
                                                         attention_encoded_context_seq, 
                                                         repeated_context_mask)
        expanded_attention_weights = attention_weights.unsqueeze(-1)
        weighted_encoded_context_seq = reshaped_encoded_context_seq * expanded_attention_weights
        weighted_encoded_context_vec = weighted_encoded_context_seq.sum(dim=1)

        # Add the last hidden state of the context vector, with the attention vector
        context_final_states = util.get_final_encoder_states(reshaped_encoded_context_seq, 
                                                             repeated_context_mask, 
                                                             bidirectional=self.context_encoder_bidirectional)
        context_final_states = self.final_hidden_state_projection_layer(context_final_states)
        weighted_encoded_context_vec = self.final_attention_projection_layer(weighted_encoded_context_vec)
        feature_vector = context_final_states + weighted_encoded_context_vec
        feature_vector = self._naive_dropout(feature_vector)

        if self.feedforward is not None:
            feature_vector = self.feedforward(feature_vector)
        logits = self.label_projection(feature_vector)
        # Reshape the logits from (Batch Size * Number target, number labels)
        # to (Batch Size, Number Targets, number labels)
        logits = logits.view(batch_size, number_targets, self.num_classes)
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
            for batch_index, sample in enumerate(metadata):
                words.append(sample['text words'])
                texts.append(sample['text'])
                targets.append(sample['targets'])
                target_words.append(sample['target words'])
                        
            output_dict["words"] = words
            output_dict["text"] = texts
            word_attention_weights = attention_weights.view(batch_size, number_targets, 
                                                            context_sequence_length)
            output_dict["word_attention"] = word_attention_weights
            output_dict["targets"] = targets
            output_dict["target words"] = target_words
            output_dict["context_mask"] = context_mask

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        '''
        Adds the predicted label to the output dict, also removes any class 
        probabilities that do not have a target associated which is caused 
        through the batch prediction process and can be removed by using the 
        target mask.

        Everyting in the dictionary will be of length (batch size * number of 
        targets) where the number of targets is based on the number of targets 
        in each sentence e.g. if the batch has two sentences where the first 
        contian 2 targets and the second 3 targets the number returned will be 
        (2 + 3) 5 target sentiments.
        '''
        batch_target_predictions = output_dict['class_probabilities'].cpu().data.numpy()
        label_masks = output_dict['targets_mask'].cpu().data.numpy()

        cpu_context_mask = output_dict["context_mask"].cpu().data.numpy()
        cpu_context_attentions_weights = output_dict["word_attention"].cpu().data.numpy()
        cpu_targets_mask = output_dict["targets_word_mask"].cpu().data.numpy()
        cpu_target_attention_weights = output_dict["targets_attention"].cpu().data.numpy()
        # Should have the same batch size and max target nubers
        batch_size = batch_target_predictions.shape[0]
        max_number_targets = batch_target_predictions.shape[1]
        assert label_masks.shape[0] == batch_size
        assert label_masks.shape[1] == max_number_targets

        _, context_sequence_length = cpu_context_mask.shape
        _, _, target_sequence_length = cpu_targets_mask.shape

        sentiments = []
        non_masked_class_probabilities = []
        word_attention = []
        target_attention = []
        for batch_index in range(batch_size):
            # Sentiment and class probabilities
            target_sentiments = []
            target_non_masked_class_probabilities = []
            target_predictions = batch_target_predictions[batch_index]
            label_mask = label_masks[batch_index]
            
            # Attention parameters
            word_attention_batch = []
            target_attention_batch = []
            relevant_word_mask = cpu_context_mask[batch_index]
            relevant_target_mask = cpu_targets_mask[batch_index]
            for target_index in range(max_number_targets):
                if label_mask[target_index] != 1:
                    continue
                # Sentiment and class probabilities
                target_prediction = target_predictions[target_index]
                label_index = numpy.argmax(target_prediction)
                label = self.vocab.get_token_from_index(label_index, 
                                                        namespace=self.label_name)
                target_sentiments.append(label)
                target_non_masked_class_probabilities.append(target_prediction)

                # Attention parameters
                context_target_word_attention = []
                for word_index in range(context_sequence_length):
                    if not relevant_word_mask[word_index]:
                        continue
                    context_target_word_attention.append(cpu_context_attentions_weights[batch_index][target_index][word_index][0])
                word_attention_batch.append(context_target_word_attention)

                target_word_attention = []
                for target_word_index in range(target_sequence_length):
                    if not relevant_target_mask[target_index][target_word_index]:
                        continue
                    target_word_attention.append(cpu_target_attention_weights[batch_index][target_index][target_word_index][0])
                target_attention_batch.append(target_word_attention)
            
            word_attention.append(word_attention_batch)
            target_attention.append(target_attention_batch)
            sentiments.append(target_sentiments)
            non_masked_class_probabilities.append(target_non_masked_class_probabilities)
        output_dict['sentiments'] = sentiments
        output_dict['class_probabilities'] = non_masked_class_probabilities
        output_dict['word_attention'] = word_attention
        output_dict['targets_attention'] = target_attention
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

        