{
    "dataset_reader": {
      "type": "target_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true,
          "token_min_padding_length": 1
        }
      },
      "incl_target": false,
      "left_right_contexts": true,
      "reverse_right_context": true
    },
    "train_data_path": "../original_target_datasets/semeval_2014/laptop_sentiment_json/train.json",
    "validation_data_path": "../original_target_datasets/semeval_2014/laptop_sentiment_json/val.json",
    "test_data_path": "../original_target_datasets/semeval_2014/laptop_sentiment_json/test.json",
    "evaluate_on_test": true,
    "model": {
      "type": "split_contexts_classifier",
      "dropout": 0.5,
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "text_field_embedder": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "../../Documents/Glove Vectors/glove.840B.300d.txt",
          "embedding_dim": 300,
          "trainable": false
        }
      },
      "left_text_encoder": {
        "type": "gru",
        "input_size": 600,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
      },
      "right_text_encoder": {
        "type": "gru",
        "input_size": 600,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
      },
      "target_encoder": {
        "type": "boe",
        "embedding_dim": 300
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 32
    },
    "trainer": {
      "optimizer": {
        "type": "adam"
      },
      "shuffle": true,
      "patience": 10,
      "num_epochs": 100,
      "cuda_device": 0,
      "validation_metric": "+accuracy"
    }
  }