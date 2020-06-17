{
    "dataset_reader": {
      "type": "target_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      },
      "left_right_contexts": true,
      "reverse_right_context": true
    },
    "train_data_path": "./tests/data/allen/models/target_sentiment/target_category_sentiments.json",
    "validation_data_path": "./tests/data/allen/models/target_sentiment/target_category_sentiments.json",
    "model": {
      "type": "split_contexts_classifier",
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
      "left_text_encoder": {
        "type": "gru",
        "input_size": 10,
        "hidden_size": 20,
        "bidirectional": true,
        "num_layers": 1
      },
      "right_text_encoder": {
        "type": "gru",
        "input_size": 10,
        "hidden_size": 20,
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
      "optimizer": {
        "type": "adam"
      },
      "num_epochs": 5,
      "grad_norm": 5.0,
      "cuda_device": -1
    }
  }