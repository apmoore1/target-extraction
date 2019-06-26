{
    "dataset_reader": {
      "type": "target_extraction",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      }
    },
    "model": {
      "type": "target_tagger",
      "label_encoding": "BIO",
      "constrain_crf_decoding": true,
      "calculate_span_f1": true,
      "dropout": 0.5,
      "include_start_end_transitions": false,
      "text_field_embedder": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "../../TDSA-Augmentation/embeddings/glove.840B.300d.txt",
          "embedding_dim": 300,
          "trainable": false
        }
      },
      "encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 50,
        "bidirectional": true,
        "num_layers": 1
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 64
    },
    "trainer": {
      "optimizer": {
        "type": "adam"
      },
      "validation_metric": "+f1-measure-overall",
      "num_epochs": 150,
      "grad_norm": 5.0,
      "patience": 10,
      "cuda_device": 0
    }
}