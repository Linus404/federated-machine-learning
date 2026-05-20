from __future__ import annotations

import os
from typing import Any

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .local_training import build_model, load_partition, sequence_length, vocab_size

# Flower ClientApp
app = ClientApp()


def load_model(embedding_dim: int = 16) -> Any:
    """Create a fresh model instance."""
    return build_model(
        vocab_size=vocab_size(),
        sequence_length=sequence_length(),
        embedding_dim=embedding_dim,
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the global model on local client data."""

    # Reset TensorFlow state between rounds
    keras.backend.clear_session()

    # Node-specific config
    partition_id = int(context.node_config["partition-id"])

    # Run config (defined in pyproject.toml or passed at runtime)
    epochs = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 64))
    embedding_dim = int(context.run_config.get("embedding-dim", 16))
    verbose = int(context.run_config.get("verbose", 1))
    validation_split = float(context.run_config.get("validation-split", 0.2))

    # Load local partition
    train_data, val_data = load_partition(
        partition=partition_id,
        validation_split=validation_split,
    )

    x_train, y_train = train_data
    x_val, y_val = val_data

    # Build model
    model = load_model(embedding_dim=embedding_dim)

    # Load global weights from server
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Local training
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Final metrics
    train_loss = float(history.history["loss"][-1])
    train_acc = float(history.history["accuracy"][-1])

    val_loss = float(history.history["val_loss"][-1])
    val_acc = float(history.history["val_accuracy"][-1])

    # Pack updated model weights
    model_record = ArrayRecord(model.get_weights())

    metrics = MetricRecord(
        {
            "num-examples": len(x_train),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metrics,
        }
    )

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on local validation data."""

    # Reset TensorFlow state
    keras.backend.clear_session()

    partition_id = int(context.node_config["partition-id"])

    embedding_dim = int(context.run_config.get("embedding-dim", 16))
    verbose = int(context.run_config.get("verbose", 0))
    validation_split = float(context.run_config.get("validation-split", 0.2))

    # Only validation data is needed
    _, val_data = load_partition(
        partition=partition_id,
        validation_split=validation_split,
    )

    x_val, y_val = val_data

    # Build model
    model = load_model(embedding_dim=embedding_dim)

    # Load global weights from server
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Evaluate
    loss, accuracy = model.evaluate(
        x_val,
        y_val,
        verbose=verbose,
    )

    metrics = MetricRecord(
        {
            "num-examples": len(x_val),
            "eval_loss": float(loss),
            "eval_accuracy": float(accuracy),
        }
    )

    content = RecordDict(
        {
            "metrics": metrics,
        }
    )

    return Message(content=content, reply_to=msg)
