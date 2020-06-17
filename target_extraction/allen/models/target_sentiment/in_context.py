from typing import Dict, Optional, List

from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy
import torch
from torch.nn.modules import Dropout, Linear
from torch.nn.functional import relu
from overrides import overrides

from target_extraction.allen.models import target_sentiment

@Model.register("in_context_classifier")
class InContextClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 target_encoding_pooling_function: str = 'mean',
                 feedforward: Optional[FeedForward] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 dropout: float = 0.0,
                 label_name: str = 'target-sentiment-labels',
                 loss_weights: Optional[List[float]] = None) -> None:
        super().__init__(vocab, regularizer)
        '''
        :param vocab: A Vocabulary, required in order to compute sizes 
                      for input/output projections.
        :param context_field_embedder: Used to embed the text and target text if
                                       target_field_embedder is None but the 
                                       target_encoder is NOT None.
        :param context_encoder: Encodes the context sentence/text.
        :param target_encoding_pooling_function: Pooling function to be used 
                                                 to create a representation 
                                                 for the target from the encoded 
                                                 context. This pooled 
                                                 representation will then be 
                                                 given to the Optional 
                                                 FeedForward layer. This can be
                                                 either `mean` for mean pooling
                                                 or `max` for max pooling. If 
                                                 this is `max` a `relu` function
                                                 is used before the pooling 
                                                 (this is to overcome the 
                                                 padding issue where some 
                                                 vectors will be zero due to 
                                                 padding.).
        :param feedforward: An optional feed forward layer to apply after the 
                            target encoding average function.
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
        
        This is based on the TD-BERT model by 
        `Gao et al. 2019 <https://ieeexplore.ieee.org/abstract/document/8864964>`_ 
        figure 2. The `target_encoding_pooling_function` when equal to `max` and the 
        `context_field_embedder` is BERT will be identical to TD-BERT.
        
        '''

        self.label_name = label_name
        self.context_field_embedder = context_field_embedder
        self.context_encoder = context_encoder
        self.num_classes = self.vocab.get_vocab_size(self.label_name)
        self.feedforward = feedforward

        allowed_pooling_functions = ['max', 'mean']
        if target_encoding_pooling_function not in allowed_pooling_functions:
            raise ValueError('Target Encoding Pooling function has to be one '
                             f'of: {allowed_pooling_functions} not: '
                             f'{target_encoding_pooling_function}')
        self.target_encoding_pooling_function = target_encoding_pooling_function 
        self.mean_pooler = BagOfEmbeddingsEncoder(self.context_encoder.get_output_dim(), 
                                                  averaged=True)
        
        # Set the loss weights (have to sort them by order of label index in 
        # the vocab)
        self.loss_weights = target_sentiment.util.loss_weight_order(self, loss_weights, self.label_name)

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = self.context_encoder.get_output_dim()
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
        check_dimensions_match(context_field_embedder.get_output_dim(),
                               context_encoder.get_input_dim(), 'Embedding',
                               'Encoder')
        if self.feedforward is not None:
            check_dimensions_match(context_encoder.get_output_dim(), 
                                   feedforward.get_input_dim(), 'Encoder', 
                                   'FeedForward')
        initializer(self)

    def forward(self, tokens: TextFieldTensors,
                targets: TextFieldTensors,
                target_sequences: torch.LongTensor,
                target_sentiments: torch.LongTensor = None,
                metadata: torch.LongTensor = None, **kwargs
                ) -> Dict[str, torch.Tensor]:
        '''
        B = Batch
        NT = Number Targets
        B_NT = Batch * Number Targets 
        TSL = Target Sequence Length
        CSL = Context Sequence Length (number tokens in the text incl padding)
        D = Dimension of the vector
        EC_D = Encoded Context Dimension
        ET_D = Embedded Text Dimension

        The text and targets are Dictionaries as they are text fields they can 
        be represented many different ways e.g. just words or words and chars 
        etc therefore the dictionary represents these different ways e.g. 
        {'words': words_tensor_ids, 'chars': char_tensor_ids}
        '''
        # A way around having targets as all they are used for is target_mask is
        # to do the following
        # target_mask = target_sequences.sum(-1) == 1
        targets_mask = util.get_text_field_mask(targets, num_wrapping_dims=1)
        b, nt, tsl = targets_mask.shape
        b_nt = b * nt

        # Embedding text and getting mask for the text/context
        text_mask = util.get_text_field_mask(tokens)
        #text_mask.names = ('B', 'CSL')
        embedded_text = self.context_field_embedder(tokens)
        embedded_text = self._variational_dropout(embedded_text)
        #embedded_text.names = ('B', 'CSL', 'ET_D')

        encoded_text = self.context_encoder(embedded_text, text_mask)
        encoded_text = self._variational_dropout(encoded_text)
        b, csl, encoded_text_dim = encoded_text.shape
        target_encoded_text = encoded_text.unsqueeze(1).repeat(1, nt, 1, 1)
        target_encoded_text = target_encoded_text.view(b_nt, csl, encoded_text_dim)
        #target_encoded_text.names = ('B_NT', 'CSL', 'EC_D')

        #target_sequences.names = ('B', 'NT', 'TSL', 'CSL')
        target_sequences = target_sequences.view(b_nt, tsl, csl)
        # Target representation that have come from the context encoder.
        encoded_targets = torch.matmul(target_sequences.type(torch.float32), 
                                       target_encoded_text)
        #encoded_targets.names = ('B_NT', 'TSL', 'EC_D')
        if self.target_encoding_pooling_function == 'max':
            encoded_targets = relu(encoded_targets)
            encoded_targets = torch.max(encoded_targets, 1)[0]
        elif self.target_encoding_pooling_function == 'mean':
            encoded_targets_mask = targets_mask.view(b_nt, tsl)
            encoded_targets = self.mean_pooler(encoded_targets, 
                                               encoded_targets_mask)
        #encoded_targets.names = ('B_NT', 'EC_D')
        encoded_targets = encoded_targets.view(b, nt, encoded_text_dim)
        if self.feedforward:
            encoded_targets = self.feedforward(encoded_targets)
        logits = self.label_projection(encoded_targets)
        label_mask = (targets_mask.sum(dim=-1) >= 1).type(torch.int64)
        # label_mask.names = ('B', 'NT')
        masked_class_probabilities = util.masked_softmax(logits, 
                                                         label_mask.unsqueeze(-1))

        output_dict = {"class_probabilities": masked_class_probabilities, 
                       "targets_mask": label_mask}
        # Convert it to bool tensor.
        label_mask = label_mask == 1

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
            meta_targets = []
            target_words = []
            for sample in metadata:
                words.append(sample['text words'])
                texts.append(sample['text'])
                meta_targets.append(sample['targets'])
                target_words.append(sample['target words'])
            output_dict["words"] = words
            output_dict["text"] = texts
            output_dict["targets"] = meta_targets
            output_dict["target words"] = target_words
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]
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