{
  "dataset_reader": {
    "type": "target_sentiment",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "target_sequences": true
  },
  "train_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
  "validation_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
  "model": {
    "type": "in_context_classifier",
    "dropout": 0.5,
    "context_field_embedder": {
      "token_embedders":{
        "tokens": {
          "type": "embedding",
          "embedding_dim": 10,
          "trainable": false
        }
      }
    },
    "context_encoder": {
      "type": "gru",
      "input_size": 10,
      "hidden_size": 12,
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