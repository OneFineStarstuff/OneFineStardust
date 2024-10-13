import logging
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorflow.keras.models import Model
import numpy as np
from typing import Optional, List, Tuple, Dict, Callable, Any
from functools import lru_cache
import asyncio
from tensorflow.keras.callbacks import TensorBoard
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

class Memory:
    def __init__(self):
        self.experiences = []

    def store_experience(self, experience):
        self.experiences.append(experience)
        if len(self.experiences) > 1000:  # Limit memory size
            self.experiences.pop(0)  # Remove oldest experience

    def retrieve_experiences(self):
        return self.experiences

class AGISystemSTEM:
    def __init__(self, model_loader: Optional[Callable] = None, distributed_training: bool = False) -> None:
        """
        Initializes the AGI system, with optional distributed training and monitoring.
        :param model_loader: Optional function for loading models, useful for dependency injection.
        :param distributed_training: Enable distributed training with TensorFlow.
        """
        self.memory = Memory()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.device = self._detect_device()
        self.model_loader = model_loader if model_loader else self._default_model_loader
        self.model_cache = {}
        # Setup distributed strategy if needed
        self.distributed_training = distributed_training
        if self.distributed_training:
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
            logging.info("Distributed training enabled with MultiWorkerMirroredStrategy.")
        else:
            self.strategy = None
        # Setup TensorBoard callback
        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    def _detect_device(self) -> str:
        """
        Detects and returns the device (GPU or CPU).
        """
        if tf.config.list_physical_devices('GPU'):
            logging.info("GPU detected.")
            return '/GPU:0'
        else:
            logging.info("Using CPU.")
            return '/CPU:0'

    @lru_cache(maxsize=5)
    async def load_model(self, model_name: str, model_class: str, version: str = "v1") -> None:
        """
        Asynchronously loads a model with a specific version.

        :param model_name: The name of the model to load.
        :param model_class: The class of the model.
        :param version: The version of the model (default is 'v1').
        """
        if model_class in self.models and version in self.models[model_class]:
            logging.info(f"{model_class} (version: {version}) already loaded.")
            return
        try:
            loaded_model = await asyncio.to_thread(self.model_loader, model_name, model_class)
            if model_class not in self.models:
                self.models[model_class] = {}
            self.models[model_class][version] = loaded_model
            logging.info(f"{model_class} version {version} loaded successfully.")
        except Exception as e:
            logging.error(f"Model load failed for version {version}: {str(e)}")

    def _default_model_loader(self, model_name: str, model_class: str) -> Any:
        """
        Default model loader function if no loader is provided.
        """
        # Example of loading a model using transformers or Keras models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model}

    def _create_dense_model(self, input_shape: Tuple[int], model_name: str, num_classes: int) -> Optional[Model]:
        """Creates a dense neural network model with optional distributed training."""
        try:
            with tf.device(self.device):
                if self.distributed_training and self.strategy:
                    with self.strategy.scope():
                        model = self._build_dense_model(input_shape, num_classes)
                else:
                    model = self._build_dense_model(input_shape, num_classes)
                logging.info(f"{model_name} model created successfully.")
                return model
        except Exception as e:
            logging.error(f"Error creating {model_name} model: {str(e)}")
            return None

    def _build_dense_model(self, input_shape: Tuple[int], num_classes: int) -> Model:
        """Helper function to build dense neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
        ])
        loss = 'categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def train_model(self, model: Model, train_data: np.ndarray, labels: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """Trains a model and logs performance with TensorBoard."""
        try:
            model.fit(
                train_data,
                labels,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[self.tensorboard_callback],
                verbose=1
            )
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")

    def evaluate_model(self, model: Model, test_data: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
        """Evaluate a model based on test data."""
        predictions = np.argmax(model.predict(test_data), axis=1)
        labels = np.argmax(labels, axis=1)
        accuracy = np.mean(predictions == labels)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        return accuracy, precision, recall, f1

    def save_model(self, model: Model, model_name: str, save_dir: str = "models") -> None:
        """Saves a trained model to disk."""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, f"{model_name}.h5")
            model.save(model_path)
            logging.info(f"Model saved at {model_path}.")
        except Exception as e:
            logging.error(f"Error saving model {model_name}: {str(e)}")

    def load_saved_model(self, model_path: str) -> Optional[Model]:
        """Loads a saved model from disk."""
        try:
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Model loaded from {model_path}.")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {str(e)}")
            return None