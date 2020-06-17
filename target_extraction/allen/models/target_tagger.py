from typing import Dict, Optional, List, Any

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

@Model.register("target_tagger")
class TargetTagger(Model):
    """
    The ``TargetTagger`` encodes a sequence of text with an optional 
    ``Seq2SeqEncoder``, then uses either Conditional Random Field 
    or simply a softmax model to predict a tag for each token in the sequence.

    This is in affect the same as the ``CrfTagger`` with the following 
    differences:
    
    1. It allows you to not have to use a ``Seq2SeqEncoder``
    2. It allows you to not have to use a ``CRF`` module but rather use a 
       the simpler softmax over the logits

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    pos_tag_embedding : ``Embedding``, optional (default=None).
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    pos_tag_loss: ``float``, optional (default=None)
        Whether to predict POS tags as an auxilary loss. The float here would 
        represent the amount to scale that loss in the overall loss function.
        The POS tags are predicted using a CRF if the main task uses a CRF else 
        like the main task it will use greedy decoding based on softmax. NOTE 
        we assume always that the label encoding for POS tags are of BIO format.
    encoder : ``Seq2SeqEncoder``, optional (default=None)
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    crf: ``bool``, optional (default=``True``)
         Whether to use a CRF, if not then it just chooses the max label over 
         the softmax (greedy decoding).
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : ``bool``, optional (default=``None``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (default=``None``). Use `Variational Dropout 
              <https://arxiv.org/abs/1512.05287>`_ for sequence and normal 
              dropout for non sequences.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pos_tag_embedding: Embedding = None,
                 pos_tag_loss: Optional[float] = None, 
                 label_namespace: str = "labels",
                 encoder: Optional[Seq2SeqEncoder] = None,
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 crf: bool = True,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        if pos_tag_loss is not None or pos_tag_embedding is not None:
            pos_tag_err = (f"Model uses POS tags but the Vocabulary {vocab} "
                           "does not contain `pos_tags` namespace")
            if 'pos_tags' not in vocab._token_to_index:
                raise ConfigurationError(pos_tag_err)
            elif not len(vocab._token_to_index['pos_tags']):
                raise ConfigurationError(pos_tag_err)
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.pos_tag_embedding = pos_tag_embedding
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics

        embedding_output_dim = self.text_field_embedder.get_output_dim()
        if self.pos_tag_embedding is not None:
            embedding_output_dim += self.pos_tag_embedding.get_output_dim()
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
            self.variational_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        elif encoder is not None:
            output_dim = self.encoder.get_output_dim()
        else:
            output_dim = embedding_output_dim
        self.tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                           self.num_tags))
        self.pos_tag_loss = pos_tag_loss
        if self.pos_tag_loss:
            self.num_pos_tags = self.vocab.get_vocab_size("pos_tags")
            self.pos_tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                                   self.num_pos_tags))
            self.pos_crf = None
            if crf:
                self.pos_crf = ConditionalRandomField(self.num_pos_tags, None,
                                                      False)


        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if crf:
            if constrain_crf_decoding is None:
                constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding and crf:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None
        if crf:
            self.include_start_end_transitions = include_start_end_transitions
            self.crf = ConditionalRandomField(
                    self.num_tags, constraints,
                    include_start_end_transitions=include_start_end_transitions
            )
        else:
            self.crf = None

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1 is not None:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                         "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=label_namespace,
                                                 label_encoding=label_encoding)
        # If performing POS tagging would be good to keep updated on POS 
        # accuracy
        if self.pos_tag_loss:
            self.metrics['POS_accuracy'] = CategoricalAccuracy()

        if encoder is not None:
            check_dimensions_match(embedding_output_dim, encoder.get_input_dim(),
                                   "text field embedding dim", "encoder input dim")
        if feedforward is not None and encoder is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        elif feedforward is not None and encoder is None:
            check_dimensions_match(embedding_output_dim, feedforward.get_input_dim(),
                                   "text field output dim", "feedforward input dim")
        initializer(self)

    def get_softmax_labels(self, class_probabilities: torch.FloatTensor,
                           mask: torch.Tensor) -> List[List[int]]:
        '''
        This method has copied a large chunck of code from the 
        `SimpleTagger.decode <https://github.com/allenai/allennlp/blob/master/allennlp/models/simple_tagger.py>`_ 
        method.
        
        Parameters
        ----------
        class_probabilities : A tensor containing the softmax scores for the 
                              tags.
        mask: A Tensor of 1's and 0's indicating whether a word exists.
        
        Returns
        -------
        A List of Lists where each inner list contains integers representing 
        the most likely tag label index based on the softmax scores. Only
        returns the tag label indexs for words that exist based on the mask
        provided. 
        '''
        
        all_predictions = class_probabilities.cpu().data.numpy()
        prediction_mask = mask.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags_indices = []
        for prediction_index, predictions in enumerate(predictions_list):
            sequence_length = prediction_mask[prediction_index].sum()
            tag_indices = numpy.argmax(predictions, axis=-1).tolist()[:sequence_length]
            all_tags_indices.append(tag_indices)
        return all_tags_indices

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor = None,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of POS tags of shape
            ``(batch_size, num_tokens)``
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence to be tagged under a 'words' key
            as well as the original text under a 'text' key.

        Returns
        -------
        An output dictionary consisting of:

        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        class_probabilities: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` 
            representing a distribution of the tag classes per word. NOTE 
            that when using the CRF the highest class probability does not 
            mean that will be the tag as that might not be globally optimal 
            for the sentence.
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[int]]``
            The predicted tags using the Viterbi algorithm if CRF is being used 
            else they are from the max over the logits using the softmax 
            approach.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        words : ``List[str]``, optional
            A list of tokens that were the original input into the model
        text : ``str``, optional
            A string that was the original text that the tokens have come from.
        """
        

        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        # Add the pos embedding
        if self.pos_tag_embedding is not None and pos_tags is not None:
            embedded_pos_tags = self.pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self.pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, "
                                     "but no POS tags were passed.")

        if self.dropout is not None:
            encoded_text = self.variational_dropout(embedded_text_input)
        
        if self.encoder is not None: 
            encoded_text = self.encoder(encoded_text, mask)
            if self.dropout is not None:
                encoded_text = self.variational_dropout(encoded_text)

        # Dropout is applied after each layer for feed forward if specified 
        # in the config.
        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        batch_size, sequence_length, _ = embedded_text_input.size()
        reshaped_log_probs = logits.view(-1, self.num_tags)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size, 
                                                                            sequence_length,
                                                                            self.num_tags])
        if self.crf is not None:
            best_paths = self.crf.viterbi_tags(logits, mask)
            # Just get the tags and ignore the score.
            predicted_tags = [x for x, y in best_paths]
        else:
            predicted_tags = self.get_softmax_labels(class_probabilities, mask)

        output = {"logits": logits, "mask": mask, "tags": predicted_tags,
                  "class_probabilities": class_probabilities}
        # Convert it to bool tensor.
        mask = mask == 1

        if tags is not None:
            # Handle either the CRF or Greedy decoder cases.
            if self.crf is not None:
                # Add negative log-likelihood as loss
                log_likelihood = self.crf(logits, tags, mask)
                output["loss"] = -log_likelihood

                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                metric_class_probabilities = logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        metric_class_probabilities[i, j, tag_id] = 1
            else:
                loss = util.sequence_cross_entropy_with_logits(logits, tags, mask)
                output["loss"] = loss
                metric_class_probabilities = logits

            for metric_name, metric in self.metrics.items():
                if metric_name != 'POS_accuracy':
                    metric(metric_class_probabilities, tags, mask)
            if self.calculate_span_f1 is not None:
                self._f1_metric(metric_class_probabilities, tags, mask)
            
            # Have to predict the POS tags to get a POS loss
            if self.pos_tag_loss and pos_tags is not None:
                pos_logits = self.pos_tag_projection_layer(encoded_text)
                # Uses the same decoding as the main task either CRF or greedy
                if self.pos_crf:
                    pos_log_likelihood = self.pos_crf(pos_logits, pos_tags, mask)
                    output["loss"] += -(self.pos_tag_loss * pos_log_likelihood)
                    # Represent viterbi tags as "class probabilities" that we can
                    # feed into the metrics
                    pos_best_paths = self.pos_crf.viterbi_tags(pos_logits, mask)
                    pos_predicted_tags = [x for x, y in pos_best_paths]
                    pos_metric_class_probabilities = pos_logits * 0.
                    for i, pos_instance_tags in enumerate(pos_predicted_tags):
                        for j, pos_tag_id in enumerate(pos_instance_tags):
                            pos_metric_class_probabilities[i, j, pos_tag_id] = 1
                else:
                    pos_loss = util.sequence_cross_entropy_with_logits(pos_logits, pos_tags, mask)
                    output["loss"] += self.pos_tag_loss * pos_loss
                    pos_metric_class_probabilities = pos_logits
                
                self.metrics['POS_accuracy'](pos_metric_class_probabilities, 
                                             pos_tags, mask.float())
            elif self.pos_tag_loss:
                raise ConfigurationError("Model uses a POS Auxilary loss, "
                                         "but no POS tags were passed.")

        if metadata is not None:
            words = []
            texts = []
            for sample in metadata:
                words.append(sample['words'])
                texts.append(sample['text'])
            output["words"] = words
            output["text"] = texts
        return output

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]
                                   ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        if self.calculate_span_f1 is not None:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in f1_dict.items() if
                        "overall" in x})
        return metrics_to_return