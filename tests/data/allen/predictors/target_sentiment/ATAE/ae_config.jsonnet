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
    "type": "atae_classifier",
    "AE": true,
    "AttentionAE": false,
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
    "context_encoder": {
      "type": "gru",
      "input_size": 17,
      "hidden_size": 10,
      "bidirectional": true,
      "num_layers": 1
    },
    "target_encoder": {
      "type": "gru",
      "input_size": 5,
      "hidden_size": 6,
      "bidirectional": true,
      "num_layers": 1
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