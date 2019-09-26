{
  "dataset_reader": {
    "type": "target_sentiment",
    "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
    }
    },
    "target_sequences": true
  },
  "train_data_path": "./tests/data/allen/dataset_readers/target_sentiment/target_sentiment_target_sequences.json",
  "validation_data_path": "./tests/data/allen/dataset_readers/target_sentiment/target_sentiment_target_sequences.json",
  "model": {
    "type": "atae_classifier",
    "dropout": 0.5,
    "use_target_sequences": true,
    "context_field_embedder": {
      "elmo": {
        "type": "language_model_token_embedder",
        "archive_file": "./tests/data/elmo_test_model/elmo_lm_test_model.tar.gz",
        "dropout": 0.2,
        "bos_eos_tokens": ["<S>", "</S>"],
        "remove_bos_eos": true,
        "requires_grad": false
      }
    },
    "context_encoder": {
      "type": "gru",
      "input_size": 44,
      "hidden_size": 10,
      "bidirectional": true,
      "num_layers": 1
    },
    "target_encoder": {
      "type": "gru",
      "input_size": 32,
      "hidden_size": 6,
      "bidirectional": true,
      "num_layers": 1
    },
    "feedforward": {
        "input_dim": 20,
        "num_layers": 1,
        "hidden_dims": 5,
        "activations": "sigmoid",
        "dropout": 0.1
      }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 1,
        "cuda_device": -1,
        "grad_clipping": 5.0,
        "validation_metric": "+accuracy",
        "optimizer": {
          "type": "adagrad"
        }
  }
}