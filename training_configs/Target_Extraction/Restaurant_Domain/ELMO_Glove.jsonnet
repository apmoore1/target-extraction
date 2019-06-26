{
    "dataset_reader": {
      "type": "target_extraction",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        },
        "elmo": {
            "type": "elmo_characters"
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
          "pretrained_file": "../../glove.6B/glove.6B.300d.txt",
          "embedding_dim": 300,
          "trainable": false
        },
        "elmo": {
            "type": "bidirectional_lm_token_embedder",
            "archive_file": "../yelp_model.tar.gz",
            "bos_eos_tokens": ["<S>", "</S>"],
            "remove_bos_eos": true,
            "requires_grad": false
          }
      },
      "encoder": {
        "type": "lstm",
        "input_size": 1324,
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