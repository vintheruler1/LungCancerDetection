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
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications import EfficientNetV2B3, ResNet152V2, DenseNet201
from tensorflow.keras import layers, regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class ImprovedLungCancerDetector:
    def __init__(self, img_size=(384, 384), batch_size=8):  # Larger image size for better detail
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.history = None
        self.class_weights = None
        
    def create_ensemble_model(self, num_classes=4):
        """Create a more powerful ensemble-like model using EfficientNetV2B3"""
        
        # Use more powerful pre-trained model
        base_model = EfficientNetV2B3(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Create more sophisticated head
        model = Sequential([
            base_model,
            
            # Global pooling with dropout
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.5),
            
            # First dense block
            Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Second dense block
            Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Third dense block
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with better optimizer settings
        initial_learning_rate = 0.001
        optimizer = Adam(
            learning_rate=initial_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'f1_score']
        )
        
        self.model = model
        print(f"Model created with {model.count_params():,} parameters")
        return model
    
    def create_advanced_data_generators(self, train_path, validation_split=0.15):
        """Create more sophisticated data augmentation"""
        
        # More aggressive but controlled augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,  # Medical images shouldn't be flipped vertically
            brightness_range=[0.8, 1.2],
            channel_shift_range=20.0,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators with better settings
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # Store class names and compute class weights for imbalanced data
        self.class_names = list(train_generator.class_indices.keys())
        
        # Calculate class weights to handle imbalanced dataset
        class_counts = []
        for class_name in self.class_names:
            class_path = os.path.join(train_path, class_name)
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts.append(count)
        
        total_samples = sum(class_counts)
        self.class_weights = {
            i: total_samples / (len(class_counts) * count) 
            for i, count in enumerate(class_counts)
        }
        
        print(f"Classes found: {self.class_names}")
        print(f"Class counts: {dict(zip(self.class_names, class_counts))}")
        print(f"Class weights: {self.class_weights}")
        
        return train_generator, val_generator
    
    def lr_schedule(self, epoch, lr):
        """Learning rate schedule"""
        if epoch < 10:
            return lr
        elif epoch < 25:
            return lr * 0.9
        elif epoch < 40:
            return lr * 0.8
        else:
            return lr * 0.7
    
    def train_improved_model(self, train_generator, val_generator, epochs=60):
        """Train with improved strategies"""
        
        # More sophisticated callbacks
        callbacks = [
            # Learning rate scheduling
            LearningRateScheduler(self.lr_schedule, verbose=1),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=7,
                factor=0.5,
                min_lr=1e-8,
                verbose=1,
                cooldown=3
            ),
            
            # Early stopping with more patience
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Model checkpointing
            ModelCheckpoint(
                'best_improved_lung_cancer_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        print("Starting initial training phase...")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        # Train model with class weights
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, train_generator, val_generator, epochs=30):
        """Fine-tune with unfrozen layers"""
        
        print("Starting fine-tuning phase...")
        
        # Unfreeze the base model gradually
        self.model.layers[0].trainable = True
        
        # Freeze early layers, unfreeze later layers
        for layer in self.model.layers[0].layers[:-50]:  # Keep first layers frozen
            layer.trainable = False
        
        # Use much lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),  # Very low learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'f1_score']
        )
        
        # Fine-tuning callbacks
        fine_tune_callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.3,
                min_lr=1e-8,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_fine_tuned_lung_cancer_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=fine_tune_callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Merge histories
        if self.history:
            for key in self.history.history:
                if key in fine_tune_history.history:
                    self.history.history[key].extend(fine_tune_history.history[key])
        
        return fine_tune_history
    
    def predict_with_confidence(self, image_path):
        """Enhanced prediction with confidence scoring"""
        # Load and preprocess image with better preprocessing
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        
        # Apply same preprocessing as training
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        confidence_scores = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = confidence_scores[predicted_class_idx] * 100
        
        # Enhanced cancer detection logic
        normal_idx = None
        for i, class_name in enumerate(self.class_names):
            if 'normal' in class_name.lower():
                normal_idx = i
                break
        
        if normal_idx is not None:
            normal_confidence = confidence_scores[normal_idx]
            cancer_confidence = (1 - normal_confidence) * 100
            is_cancer = normal_confidence < 0.5  # More conservative threshold
        else:
            cancer_confidence = confidence
            is_cancer = predicted_class != 'normal'
        
        # Calculate uncertainty (entropy)
        entropy = -np.sum(confidence_scores * np.log(confidence_scores + 1e-8))
        uncertainty = entropy / np.log(len(confidence_scores))  # Normalized
        
        results = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_confidence': cancer_confidence,
            'uncertainty': uncertainty * 100,
            'all_probabilities': {
                self.class_names[i]: prob * 100 
                for i, prob in enumerate(confidence_scores)
            }
        }
        
        return results
    
    def evaluate_comprehensive(self, test_path):
        """Comprehensive evaluation on test set"""
        print("Running comprehensive evaluation...")
        
        results = []
        true_labels = []
        pred_labels = []
        
        for class_folder in os.listdir(test_path):
            class_path = os.path.join(test_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            print(f"Evaluating {class_folder}...")
            
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        prediction_result = self.predict_with_confidence(image_path)
                        
                        # Map folder names to class names
                        true_class = class_folder
                        for i, class_name in enumerate(self.class_names):
                            if class_name.split('_')[0] in true_class or true_class in class_name:
                                true_class = class_name
                                break
                        
                        results.append({
                            'image_path': image_path,
                            'true_class': true_class,
                            'predicted_class': prediction_result['predicted_class'],
                            'confidence': prediction_result['confidence'],
                            'uncertainty': prediction_result['uncertainty'],
                            'is_cancer': prediction_result['is_cancer'],
                            'cancer_confidence': prediction_result['cancer_confidence']
                        })
                        
                        true_labels.append(true_class)
                        pred_labels.append(prediction_result['predicted_class'])
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        
        # Create comprehensive report
        df_results = pd.DataFrame(results)
        
        # Calculate accuracy metrics
        accuracy = (df_results['true_class'] == df_results['predicted_class']).mean()
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Total Images Tested: {len(df_results)}")
        
        # Per-class accuracy
        print(f"\nPer-Class Accuracy:")
        for true_class in df_results['true_class'].unique():
            class_df = df_results[df_results['true_class'] == true_class]
            class_acc = (class_df['true_class'] == class_df['predicted_class']).mean()
            print(f"  {true_class.split('_')[0]}: {class_acc:.3f} ({class_acc*100:.1f}%)")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        unique_classes = sorted(list(set(true_labels + pred_labels)))
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_classes)
        cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
        print(cm_df)
        
        return df_results, accuracy
    
    def create_activation_heatmap(self, image_path, layer_name=None):
        """Create heatmap showing areas the model focuses on"""
        # Load and preprocess image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get the correct layer from EfficientNetV2B3
        if layer_name is None:
            # Use the last convolutional layer from EfficientNetV2B3 base model
            base_model = self.model.layers[0]  # The EfficientNetV2B3 base
            
            # Find the last convolutional layer in the base model
            conv_layers = []
            for layer in base_model.layers:
                if 'conv' in layer.name.lower() and hasattr(layer, 'activation'):
                    conv_layers.append(layer)
            
            if conv_layers:
                layer_name = conv_layers[-1].name
                print(f"Using layer: {layer_name}")
            else:
                # Fallback: use a known layer from EfficientNetV2B3
                layer_name = 'top_activation'  # Common final activation layer
                # If that doesn't exist, try finding any activation layer
                for layer in base_model.layers:
                    if 'activation' in layer.name or 'relu' in layer.name:
                        layer_name = layer.name
                        break
        
        try:
            # Create model that outputs the activations of the specified layer
            layer_output = base_model.get_layer(layer_name).output
            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[layer_output, self.model.output]
            )
            
            # Compute gradients using GradCAM
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
            
        except Exception as e:
            print(f"Error with layer {layer_name}: {e}")
            print("Available base model layers:")
            for i, layer in enumerate(base_model.layers[-10:]):  # Show last 10 layers
                print(f"  {i}: {layer.name} ({type(layer).__name__})")
            
            # Try a simpler approach - use global average pooling layer
            try:
                # Use the feature maps just before global average pooling
                for layer in reversed(base_model.layers):
                    if len(layer.output_shape) == 4:  # 4D tensor (batch, height, width, channels)
                        layer_name = layer.name
                        print(f"Fallback to layer: {layer_name}")
                        break
                
                layer_output = base_model.get_layer(layer_name).output
                grad_model = Model(
                    inputs=self.model.inputs,
                    outputs=[layer_output, self.model.output]
                )
                
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    predicted_class_idx = tf.argmax(predictions[0])
                    class_score = predictions[:, predicted_class_idx]
                
                grads = tape.gradient(class_score, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                
                return heatmap.numpy()
                
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return None
    
    def highlight_cancer_spots(self, image_path, output_path=None, alpha=0.6):
        """Highlight potential cancer spots with improved circling"""
        # Get heatmap
        heatmap = self.create_activation_heatmap(image_path)
        if heatmap is None:
            print("Could not generate heatmap - creating simple overlay")
            return None, None
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, self.img_size)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, self.img_size)
        
        # Convert heatmap to RGB and apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Create overlay
        overlayed_img = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
        
        # Find and circle potential cancer spots with improved detection
        # Use multiple thresholds for better spot detection
        high_threshold = np.percentile(heatmap_resized, 90)  # Top 10%
        med_threshold = np.percentile(heatmap_resized, 80)   # Top 20%
        
        # Find high activation areas
        high_spots = np.where(heatmap_resized > high_threshold)
        med_spots = np.where(heatmap_resized > med_threshold)
        
        # Group nearby points into clusters (cancer regions)
        def find_clusters(spots_y, spots_x, min_distance=20):
            if len(spots_y) == 0:
                return []
            
            points = list(zip(spots_y, spots_x))
            clusters = []
            
            for point in points:
                added_to_cluster = False
                for cluster in clusters:
                    # Check if point is close to any point in existing cluster
                    for cluster_point in cluster:
                        distance = np.sqrt((point[0] - cluster_point[0])**2 + (point[1] - cluster_point[1])**2)
                        if distance < min_distance:
                            cluster.append(point)
                            added_to_cluster = True
                            break
                    if added_to_cluster:
                        break
                
                if not added_to_cluster:
                    clusters.append([point])
            
            return clusters
        
        # Find clusters of high activation areas
        high_clusters = find_clusters(high_spots[0], high_spots[1], min_distance=25)
        med_clusters = find_clusters(med_spots[0], med_spots[1], min_distance=30)
        
        # Draw circles around high activation clusters (bright red)
        for cluster in high_clusters:
            if len(cluster) > 3:  # Only mark significant clusters
                # Find center of cluster
                center_y = int(np.mean([p[0] for p in cluster]))
                center_x = int(np.mean([p[1] for p in cluster]))
                
                # Calculate radius based on cluster size
                max_dist = max([np.sqrt((p[0] - center_y)**2 + (p[1] - center_x)**2) for p in cluster])
                radius = int(max(max_dist + 10, 15))  # Minimum radius of 15
                
                # Draw thick red circle for high confidence areas
                cv2.circle(overlayed_img, (center_x, center_y), radius, (0, 0, 255), 4)  # Red circle
                cv2.circle(overlayed_img, (center_x, center_y), 3, (0, 0, 255), -1)  # Red center dot
                
                # Add text label
                cv2.putText(overlayed_img, 'HIGH', (center_x - 20, center_y - radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw circles around medium activation clusters (orange)
        for cluster in med_clusters:
            if len(cluster) > 5:  # Need more points for medium confidence
                center_y = int(np.mean([p[0] for p in cluster]))
                center_x = int(np.mean([p[1] for p in cluster]))
                
                # Skip if too close to a high confidence area
                too_close = False
                for high_cluster in high_clusters:
                    if len(high_cluster) > 3:
                        high_center_y = int(np.mean([p[0] for p in high_cluster]))
                        high_center_x = int(np.mean([p[1] for p in high_cluster]))
                        distance = np.sqrt((center_y - high_center_y)**2 + (center_x - high_center_x)**2)
                        if distance < 40:
                            too_close = True
                            break
                
                if not too_close:
                    max_dist = max([np.sqrt((p[0] - center_y)**2 + (p[1] - center_x)**2) for p in cluster])
                    radius = int(max(max_dist + 8, 12))
                    
                    # Draw thinner orange circle for medium confidence
                    cv2.circle(overlayed_img, (center_x, center_y), radius, (0, 165, 255), 3)  # Orange circle
                    cv2.circle(overlayed_img, (center_x, center_y), 2, (0, 165, 255), -1)  # Orange center dot
                    
                    cv2.putText(overlayed_img, 'MED', (center_x - 15, center_y - radius - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Add legend
        legend_y = 30
        cv2.rectangle(overlayed_img, (10, 10), (200, 80), (0, 0, 0), -1)  # Black background
        cv2.rectangle(overlayed_img, (10, 10), (200, 80), (255, 255, 255), 2)  # White border
        
        cv2.circle(overlayed_img, (25, legend_y), 8, (0, 0, 255), 3)
        cv2.putText(overlayed_img, 'High Confidence', (40, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(overlayed_img, (25, legend_y + 25), 6, (0, 165, 255), 3)
        cv2.putText(overlayed_img, 'Medium Confidence', (40, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, overlayed_img)
        
        return overlayed_img, heatmap_resized
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[0, 2].plot(self.history.history['precision'], label='Training', linewidth=2)
            axes[0, 2].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
            axes[0, 2].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 0].plot(self.history.history['recall'], label='Training', linewidth=2)
            axes[1, 0].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        if 'f1_score' in self.history.history:
            axes[1, 1].plot(self.history.history['f1_score'], label='Training', linewidth=2)
            axes[1, 1].plot(self.history.history['val_f1_score'], label='Validation', linewidth=2)
            axes[1, 1].set_title('Model F1-Score', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr_values = []
            for epoch in range(len(self.history.history['loss'])):
                lr_values.append(self.lr_schedule(epoch, 0.001))
            axes[1, 2].plot(lr_values, linewidth=2, color='red')
            axes[1, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
            
            # Also save class names and weights
            import pickle
            metadata = {
                'class_names': self.class_names,
                'class_weights': self.class_weights,
                'img_size': self.img_size
            }
            with open(filepath.replace('.h5', '_metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Model metadata saved to {filepath.replace('.h5', '_metadata.pkl')}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Try to load metadata
        try:
            import pickle
            with open(filepath.replace('.h5', '_metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
                self.class_names = metadata['class_names']
                self.class_weights = metadata['class_weights']
                self.img_size = metadata['img_size']
            print("Model metadata loaded successfully!")
        except:
            print("No metadata file found. Please set class_names manually.")

def main():
    """Enhanced training script"""
    print("🔬 IMPROVED Lung Cancer Detection System")
    print("=" * 60)
    
    # Use larger image size and smaller batch size for better quality
    detector = ImprovedLungCancerDetector(img_size=(384, 384), batch_size=8)
    
    # Create improved model
    model = detector.create_ensemble_model(num_classes=4)
    print("Enhanced model created!")
    
    # Prepare data
    train_path = "dataset/train"
    
    if os.path.exists(train_path):
        print("Preparing enhanced training data...")
        train_gen, val_gen = detector.create_advanced_data_generators(train_path)
        
        # Train with improved strategy
        print("Starting enhanced training...")
        history = detector.train_improved_model(train_gen, val_gen, epochs=50)
        
        # Fine-tune for even better performance
        print("Fine-tuning for maximum accuracy...")
        detector.fine_tune_model(train_gen, val_gen, epochs=25)
        
        # Evaluate on test set
        test_path = "dataset/test"
        if os.path.exists(test_path):
            results_df, final_accuracy = detector.evaluate_comprehensive(test_path)
            print(f"\n🎯 FINAL ACCURACY: {final_accuracy*100:.2f}%")
            
            if final_accuracy > 0.85:
                print("🎉 Excellent performance achieved!")
            elif final_accuracy > 0.75:
                print("✅ Good performance achieved!")
            else:
                print("⚠️ Consider more training or data augmentation")
        
        # Save final model
        detector.save_model("improved_lung_cancer_detector.h5")
        print("Model saved successfully!")
        
    else:
        print(f"Training path {train_path} not found!")

if __name__ == "__main__":
    main()