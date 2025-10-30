import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class LungCancerDetector:
    def __init__(self, img_size=(224, 224), batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.history = None
        
    def create_model(self, num_classes=4):
        """Create an advanced CNN model using EfficientNet for better accuracy"""
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, train_path, test_path=None, validation_split=0.2):
        """Prepare data generators for training"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.1,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation/test
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes found: {self.class_names}")
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=50):
        """Train the model with callbacks"""
        # Callbacks
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.5,
                min_lr=0.00001,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lung_cancer_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, train_generator, val_generator, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        # Unfreeze the top layers of the base model
        self.model.layers[0].trainable = True
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Fine-tuning model...")
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        return fine_tune_history
    
    def predict_with_confidence(self, image_path):
        """Predict cancer type with confidence percentage"""
        # Load and preprocess image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        confidence_scores = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = confidence_scores[predicted_class_idx] * 100
        
        # Check if it's cancer (not normal)
        is_cancer = predicted_class != 'normal'
        cancer_confidence = (1 - confidence_scores[self.class_names.index('normal')]) * 100 if 'normal' in self.class_names else confidence
        
        results = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_confidence': cancer_confidence,
            'all_probabilities': {class_name: prob * 100 for class_name, prob in zip(self.class_names, confidence_scores)}
        }
        
        return results
    
    def create_activation_heatmap(self, image_path, layer_name=None):
        """Create heatmap showing areas the model focuses on"""
        # Load and preprocess image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get the last convolutional layer if no layer specified
        if layer_name is None:
            conv_layers = [layer for layer in self.model.layers[0].layers if isinstance(layer, Conv2D)]
            if conv_layers:
                layer_name = conv_layers[-1].name
            else:
                print("No convolutional layers found")
                return None
        
        # Create model that outputs the activations of the specified layer
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, predicted_class_idx]
        
        # Get gradients of the class score with respect to feature map
        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature map by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def highlight_cancer_spots(self, image_path, output_path=None, alpha=0.6):
        """Highlight potential cancer spots on the image"""
        # Get heatmap
        heatmap = self.create_activation_heatmap(image_path)
        if heatmap is None:
            return None
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, self.img_size)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, self.img_size)
        
        # Convert heatmap to RGB and apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlayed_img = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
        
        # Find potential cancer spots (high activation areas)
        threshold = np.percentile(heatmap_resized, 85)  # Top 15% of activations
        cancer_spots = np.where(heatmap_resized > threshold)
        
        if len(cancer_spots[0]) > 0:
            # Draw circles around high activation areas
            for y, x in zip(cancer_spots[0][::10], cancer_spots[1][::10]):  # Sample every 10th point
                cv2.circle(overlayed_img, (x, y), 5, (0, 255, 0), 2)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, overlayed_img)
        
        return overlayed_img, heatmap_resized
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_test_images(self, test_path):
        """Evaluate model on test images and show detailed results"""
        results = []
        
        for class_folder in os.listdir(test_path):
            class_path = os.path.join(test_path, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    
                    # Make prediction
                    prediction_result = self.predict_with_confidence(image_path)
                    
                    results.append({
                        'image_path': image_path,
                        'true_class': class_folder,
                        'predicted_class': prediction_result['predicted_class'],
                        'confidence': prediction_result['confidence'],
                        'is_cancer': prediction_result['is_cancer'],
                        'cancer_confidence': prediction_result['cancer_confidence']
                    })
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and training script
def main():
    print("🔬 Lung Cancer Detection and Classification System")
    print("=" * 50)
    
    # Initialize detector
    detector = LungCancerDetector(img_size=(224, 224), batch_size=16)
    
    # Create model
    model = detector.create_model(num_classes=4)
    print("Model created successfully!")
    
    # Prepare data (adjust paths as needed)
    train_path = "dataset/train"
    
    if os.path.exists(train_path):
        print("Preparing training data...")
        train_gen, val_gen = detector.prepare_data(train_path)
        
        # Train model
        print("Starting model training...")
        history = detector.train_model(train_gen, val_gen, epochs=30)
        
        # Fine-tune model
        print("Fine-tuning model...")
        detector.fine_tune_model(train_gen, val_gen, epochs=10)
        
        # Plot training history
        detector.plot_training_history()
        
        # Save model
        detector.save_model("lung_cancer_detector.h5")
        
        # Evaluate on test data
        test_path = "dataset/test"
        if os.path.exists(test_path):
            print("Evaluating on test data...")
            test_results = detector.evaluate_test_images(test_path)
            print("\nTest Results Summary:")
            print(test_results.groupby(['true_class', 'predicted_class']).size().unstack(fill_value=0))
    
    else:
        print(f"Training path {train_path} not found!")
        print("Please make sure your dataset is properly organized.")

if __name__ == "__main__":
    main()