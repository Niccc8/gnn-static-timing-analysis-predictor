# tests/

This directory is reserved for unit tests and integration tests.

## Structure (To Be Implemented)

```
tests/
├── test_dataset.py           # Dataset loading tests
├── test_feature_extractor.py  # Feature extraction tests
├── test_graph_builder.py      # Graph construction tests
├── test_model.py              # Model forward pass tests
└── test_training.py           # Training loop tests
```

## Running Tests

```bash
pytest tests/ -v
```

## Future Work

- Add unit tests for all core modules
- Add integration tests for full pipeline
- Add CI/CD with GitHub Actions
