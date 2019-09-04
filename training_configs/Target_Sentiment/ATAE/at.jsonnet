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
  "train_data_path": "../original_target_datasets/semeval_2014/laptop_sentiment_json/train.json",
  "validation_data_path": "../original_target_datasets/semeval_2014/laptop_sentiment_json/val.json",
  "test_data_path": "../original_target_datasets/semeval_2014/laptop_sentiment_json/test.json",
  "evaluate_on_test": true,
  "model": {
    "type": "atae_classifier",
    "dropout": 0.5,
    "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
    "AE": false,
    "AttentionAE": true,
    "context_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 300,
        "pretrained_file": "../../Documents/Glove Vectors/glove.840B.300d.txt",
        "trainable": false
      }
    },
    "context_encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
    },
    "target_encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
    },
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