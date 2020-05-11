{
  "dataset_reader": {
    "type": "target_sentiment",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "train_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
  "validation_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
  "model": {
    "type": "interactive_attention_network_classifier",
    "dropout": 0.5,
    "context_field_embedder": {
      "token_embedders":{
        "tokens": {
          "type": "embedding",
          "embedding_dim": 5,
          "trainable": false
        }
      }
    },
    "target_encoder": {
      "type": "gru",
      "input_size": 5,
      "hidden_size": 5,
      "bidirectional": false,
      "num_layers": 1
    },
    "context_encoder": {
      "type": "gru",
      "input_size": 5,
      "hidden_size": 5,
      "bidirectional": false,
      "num_layers": 1
    },
    "feedforward": {
        "input_dim": 10,
        "num_layers": 1,
        "hidden_dims": 4,
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