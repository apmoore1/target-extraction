{
  "dataset_reader": {
    "type": "target_sentiment",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "elmo": {
        "type": "custom_elmo_characters"
    }
    }
  },
  "train_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
  "validation_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
  "model": {
    "type": "atae_classifier",
    "dropout": 0.5,
    "context_field_embedder": {
      "token_embedders":
      {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 5,
          "trainable": false
        },
        "elmo": {
          "type": "language_model_token_embedder",
          "archive_file": "./tests/data/elmo_test_model/elmo_lm_test_model.tar.gz",
          "dropout": 0.2,
          "bos_eos_tokens": ["<S>", "</S>"],
          "remove_bos_eos": true,
          "requires_grad": false
        }
      }
    },
    "context_encoder": {
      "type": "gru",
      "input_size": 49,
      "hidden_size": 10,
      "bidirectional": true,
      "num_layers": 1
    },
    "target_encoder": {
      "type": "gru",
      "input_size": 37,
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
  "data_loader": {
    "batch_size": 64,
    "shuffle": true,
    "drop_last": false
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