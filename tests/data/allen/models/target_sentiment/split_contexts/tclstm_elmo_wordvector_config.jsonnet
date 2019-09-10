{
    "dataset_reader": {
      "type": "target_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        },
      "elmo": {
        "type": "elmo_characters"
      }
      },
      "left_right_contexts": true,
      "reverse_right_context": true
    },
    "train_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
    "validation_data_path": "./tests/data/allen/models/target_sentiment/multi_target_category_sentiments.json",
    "model": {
      "type": "split_contexts_classifier",
      "dropout": 0.5,
      "context_field_embedder": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 10,
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
      },
      "left_text_encoder": {
        "type": "gru",
        "input_size": 84,
        "hidden_size": 20,
        "bidirectional": true,
        "num_layers": 1
      },
      "right_text_encoder": {
        "type": "gru",
        "input_size": 84,
        "hidden_size": 20,
        "bidirectional": true,
        "num_layers": 1
      },
      "target_encoder": {
        "type": "boe",
        "embedding_dim": 42,
        "averaged": true
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
      "num_epochs": 5,
      "grad_norm": 5.0,
      "cuda_device": -1
    }
  }