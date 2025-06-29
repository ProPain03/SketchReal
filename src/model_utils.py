# utils/model_utils.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from config.config import Config

# Add this function at the beginning of the file:
def fix_torch_classes_issue():
    """
    Fix for PyTorch's custom classes issue with Streamlit's file watcher
    """
    import sys
    
    # If torch is already imported, we need to handle the _classes module
    if 'torch' in sys.modules:
        import torch
        if hasattr(torch, '_classes'):
            # Override __getattr__ to prevent Streamlit's watcher from accessing __path__
            original_getattr = torch._classes.__getattr__
            
            def safe_getattr(name):
                if name == '__path__':
                    # Return an empty list instead of accessing _path
                    return []
                return original_getattr(name)
            
            # Apply the patched __getattr__
            torch._classes.__getattr__ = safe_getattr

# Call this function immediately
fix_torch_classes_issue()

# Rest of your ModelInterface code...

class SketchCNNModel:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.history = None
    
    def build_baseline_cnn(self):
        """Build a simple CNN model for sketch classification"""
        self.model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=None):
        """Compile the model with optimizer and loss function"""
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        return self.model
    
    def get_callbacks(self, model_save_path):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=None):
        """Train the CNN model"""
        if epochs is None:
            epochs = self.config.EPOCHS
            
        # Ensure model directory exists
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        model_save_path = os.path.join(self.config.MODEL_DIR, 'baseline_model.h5')
        
        # Get callbacks
        callbacks = self.get_callbacks(model_save_path)
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get predictions
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        test_loss = evaluation[0]
        test_accuracy = evaluation[1]
        test_top_3 = evaluation[2] if len(evaluation) > 2 else 0
    
        # Get detailed predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
        # Calculate top-3 accuracy manually
        top_3_accuracy = self.calculate_top_k_accuracy(y_test, y_pred_proba, k=3)
    
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'top_3_accuracy': top_3_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
        return results
    
    def calculate_top_k_accuracy(self, y_true, y_pred_proba, k=3):
        """Calculate top-k accuracy"""
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        return correct / len(y_true)
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available!")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names if class_names else range(len(cm)),
                   yticklabels=class_names if class_names else range(len(cm)))
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, class_names=None):
        """Generate detailed classification report"""
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.config.NUM_CLASSES)]
            
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Print formatted report
        print("Classification Report:")
        print("=" * 50)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        return report
    
    def save_model(self, save_path=None):
        """Save the trained model"""
        if save_path is None:
            save_path = os.path.join(self.config.MODEL_DIR, 'final_model.h5')
            
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict_single_image(self, image, class_names=None):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded!")
            
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        if class_names:
            predicted_label = class_names[predicted_class]
            return predicted_label, confidence
        else:
            return predicted_class, confidence
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise ValueError("Model not built yet!")
            
        self.model.summary()
        
        # Calculate total parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        return total_params