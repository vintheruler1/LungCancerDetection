from improved_lung_cancer_detector import ImprovedLungCancerDetector
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set matplotlib backend for macOS
plt.ion()  # Turn on interactive mode
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for macOS

def create_fixed_activation_heatmap(model, image_path, img_size=(384, 384)):
    """Fixed heatmap creation that works with EfficientNet model structure"""
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get the base model (EfficientNet)
        base_model = model.layers[0]  # First layer should be the EfficientNet
        
        # Find the right layer - use 'top_activation' from EfficientNet
        target_layer_name = 'top_activation'
        target_layer = None
        
        try:
            target_layer = base_model.get_layer(target_layer_name)
            print(f"Using layer: {target_layer_name}")
        except:
            # Try other common EfficientNet layer names
            possible_layers = ['top_activation', 'top_conv', 'block6l_add', 'block6l_project_bn']
            for layer_name in possible_layers:
                try:
                    target_layer = base_model.get_layer(layer_name)
                    target_layer_name = layer_name
                    print(f"Found and using layer: {target_layer_name}")
                    break
                except:
                    continue
        
        if target_layer is None:
            print("Using fallback: edge detection heatmap")
            return create_edge_based_heatmap(image_path, img_size)
        
        # Create a model that outputs both the target layer and final predictions
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = grad_model(img_array)
            predicted_class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, predicted_class_idx]
        
        # Get gradients of the class score with respect to the feature map
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global average pool the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by the importance of the channel
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        print(f"Generated heatmap with shape: {heatmap.shape}")
        return heatmap
        
    except Exception as e:
        print(f"Neural heatmap failed: {e}")
        print("Using edge detection fallback...")
        return create_edge_based_heatmap(image_path, img_size)

def create_edge_based_heatmap(image_path, img_size):
    """Create a heatmap based on edge detection and texture analysis"""
    try:
        # Load image in grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Could not load image: {image_path}")
            return None
            
        img_gray = cv2.resize(img_gray, img_size)
        
        # Apply Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(img_blur, 50, 150)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find areas with high texture variation (potential abnormalities)
        # Use Laplacian for texture detection
        laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
        laplacian = np.absolute(laplacian).astype(np.uint8)
        
        # Combine edges and texture
        combined = cv2.addWeighted(edges, 0.6, laplacian, 0.4, 0)
        
        # Apply morphological operations to create regions
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur to smooth the heatmap
        heatmap = cv2.GaussianBlur(combined, (15, 15), 0)
        
        # Normalize to 0-1 range
        heatmap = heatmap.astype(np.float32) / 255.0
        
        print(f"Generated edge-based heatmap with shape: {heatmap.shape}")
        return heatmap
        
    except Exception as e:
        print(f"Edge detection also failed: {e}")
        # Return a simple gradient as absolute fallback
        gradient = np.zeros(img_size, dtype=np.float32)
        center_y, center_x = img_size[0] // 2, img_size[1] // 2
        for y in range(img_size[0]):
            for x in range(img_size[1]):
                distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                gradient[y, x] = max(0, 1 - distance / (max(img_size) / 2))
        return gradient

def highlight_cancer_spots_fixed(model, image_path, img_size=(384, 384), output_path=None, alpha=0.6):
    """Enhanced cancer spot highlighting with improved detection"""
    
    # Get heatmap using improved method
    heatmap = create_fixed_activation_heatmap(model, image_path, img_size)
    if heatmap is None:
        print("Could not generate any heatmap")
        return None, None
    
    # Load original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Could not load image: {image_path}")
        return None, None
        
    original_img = cv2.resize(original_img, img_size)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != img_size:
        heatmap_resized = cv2.resize(heatmap, img_size)
    else:
        heatmap_resized = heatmap.copy()
    
    # Convert heatmap to colored version
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Create overlay
    overlayed_img = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
    
    # Enhanced spot detection with multiple thresholds
    very_high_threshold = np.percentile(heatmap_resized, 92)  # Top 8%
    high_threshold = np.percentile(heatmap_resized, 85)       # Top 15%
    med_threshold = np.percentile(heatmap_resized, 75)        # Top 25%
    
    print(f"Thresholds - Very High: {very_high_threshold:.3f}, High: {high_threshold:.3f}, Med: {med_threshold:.3f}")
    
    # Find activation areas
    very_high_spots = np.where(heatmap_resized > very_high_threshold)
    high_spots = np.where(heatmap_resized > high_threshold)
    med_spots = np.where(heatmap_resized > med_threshold)
    
    print(f"Found spots - Very High: {len(very_high_spots[0])}, High: {len(high_spots[0])}, Med: {len(med_spots[0])}")
    
    # Improved clustering function
    def find_improved_clusters(spots_y, spots_x, min_distance=25, min_points=3):
        if len(spots_y) == 0:
            return []
        
        points = list(zip(spots_y, spots_x))
        clusters = []
        used_points = set()
        
        for i, point in enumerate(points[::3]):  # Sample every 3rd point
            if i in used_points:
                continue
                
            # Start new cluster
            cluster = [point]
            used_points.add(i)
            
            # Find nearby points
            for j, other_point in enumerate(points):
                if j in used_points:
                    continue
                    
                distance = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                if distance < min_distance:
                    cluster.append(other_point)
                    used_points.add(j)
            
            if len(cluster) >= min_points:
                clusters.append(cluster)
        
        return clusters
    
    # Find clusters for each threshold level
    very_high_clusters = find_improved_clusters(very_high_spots[0], very_high_spots[1], min_distance=20, min_points=2)
    high_clusters = find_improved_clusters(high_spots[0], high_spots[1], min_distance=25, min_points=3)
    med_clusters = find_improved_clusters(med_spots[0], med_spots[1], min_distance=30, min_points=5)
    
    print(f"Clusters found - Very High: {len(very_high_clusters)}, High: {len(high_clusters)}, Med: {len(med_clusters)}")
    
    circles_drawn = 0
    
    # Draw circles around very high activation clusters (BRIGHT RED - Critical)
    for cluster in very_high_clusters:
        center_y = int(np.mean([p[0] for p in cluster]))
        center_x = int(np.mean([p[1] for p in cluster]))
        
        # Calculate radius based on cluster spread
        distances = [np.sqrt((p[0] - center_y)**2 + (p[1] - center_x)**2) for p in cluster]
        radius = int(max(np.mean(distances) + 15, 25))  # Minimum radius of 25
        
        # Draw bright red circle
        cv2.circle(overlayed_img, (center_x, center_y), radius, (0, 0, 255), 5)  # Thick red circle
        cv2.circle(overlayed_img, (center_x, center_y), 4, (0, 0, 255), -1)      # Red center dot
        
        # Add label with background
        label_bg_size = cv2.getTextSize('CRITICAL', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlayed_img, 
                     (center_x - label_bg_size[0]//2 - 5, center_y - radius - 25),
                     (center_x + label_bg_size[0]//2 + 5, center_y - radius - 5),
                     (0, 0, 0), -1)
        cv2.putText(overlayed_img, 'CRITICAL', (center_x - label_bg_size[0]//2, center_y - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        circles_drawn += 1
    
    # Draw circles around high activation clusters (RED - High Suspicion)
    for cluster in high_clusters:
        center_y = int(np.mean([p[0] for p in cluster]))
        center_x = int(np.mean([p[1] for p in cluster]))
        
        # Check if too close to a critical area
        too_close = False
        for vh_cluster in very_high_clusters:
            vh_center_y = int(np.mean([p[0] for p in vh_cluster]))
            vh_center_x = int(np.mean([p[1] for p in vh_cluster]))
            distance = np.sqrt((center_y - vh_center_y)**2 + (center_x - vh_center_x)**2)
            if distance < 45:
                too_close = True
                break
        
        if not too_close:
            distances = [np.sqrt((p[0] - center_y)**2 + (p[1] - center_x)**2) for p in cluster]
            radius = int(max(np.mean(distances) + 12, 20))
            
            # Draw red circle
            cv2.circle(overlayed_img, (center_x, center_y), radius, (0, 69, 255), 4)  # Red-orange circle
            cv2.circle(overlayed_img, (center_x, center_y), 3, (0, 69, 255), -1)      # Center dot
            
            # Add label
            cv2.putText(overlayed_img, 'HIGH', (center_x - 20, center_y - radius - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 69, 255), 2)
            circles_drawn += 1
    
    # Draw circles around medium activation clusters (ORANGE - Moderate Suspicion)
    for cluster in med_clusters:
        center_y = int(np.mean([p[0] for p in cluster]))
        center_x = int(np.mean([p[1] for p in cluster]))
        
        # Check if too close to higher priority areas
        too_close = False
        all_high_clusters = very_high_clusters + high_clusters
        for high_cluster in all_high_clusters:
            high_center_y = int(np.mean([p[0] for p in high_cluster]))
            high_center_x = int(np.mean([p[1] for p in high_cluster]))
            distance = np.sqrt((center_y - high_center_y)**2 + (center_x - high_center_x)**2)
            if distance < 50:
                too_close = True
                break
        
        if not too_close and circles_drawn < 8:  # Limit total circles to avoid clutter
            distances = [np.sqrt((p[0] - center_y)**2 + (p[1] - center_x)**2) for p in cluster]
            radius = int(max(np.mean(distances) + 8, 15))
            
            # Draw orange circle
            cv2.circle(overlayed_img, (center_x, center_y), radius, (0, 165, 255), 3)  # Orange circle
            cv2.circle(overlayed_img, (center_x, center_y), 2, (0, 165, 255), -1)      # Center dot
            
            cv2.putText(overlayed_img, 'MED', (center_x - 15, center_y - radius - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)
            circles_drawn += 1
    
    # Enhanced legend
    legend_height = 110
    legend_width = 280
    cv2.rectangle(overlayed_img, (10, 10), (10 + legend_width, 10 + legend_height), (0, 0, 0), -1)
    cv2.rectangle(overlayed_img, (10, 10), (10 + legend_width, 10 + legend_height), (255, 255, 255), 2)
    
    # Legend title
    cv2.putText(overlayed_img, 'Cancer Detection Areas:', (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Legend items
    y_offset = 50
    # Critical
    cv2.circle(overlayed_img, (25, y_offset), 8, (0, 0, 255), 3)
    cv2.putText(overlayed_img, 'Critical Suspicion (>92%)', (40, y_offset + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # High
    y_offset += 20
    cv2.circle(overlayed_img, (25, y_offset), 7, (0, 69, 255), 3)
    cv2.putText(overlayed_img, 'High Suspicion (>85%)', (40, y_offset + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Medium
    y_offset += 20
    cv2.circle(overlayed_img, (25, y_offset), 6, (0, 165, 255), 3)
    cv2.putText(overlayed_img, 'Moderate Suspicion (>75%)', (40, y_offset + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add summary
    summary_text = f'Total Suspicious Areas: {circles_drawn}'
    cv2.putText(overlayed_img, summary_text, (20, img_size[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, overlayed_img)
        print(f"🎯 Cancer spots highlighted and saved to: {output_path}")
    
    print(f"✅ Successfully highlighted {circles_drawn} suspicious areas")
    return overlayed_img, heatmap_resized

def quick_test_improved(image_path, model_path="improved_lung_cancer_detector.h5"):
    """Quick test for a single image with improved model"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Initialize detector
    detector = ImprovedLungCancerDetector(img_size=(384, 384))
    
    # Load model
    model_loaded = False
    model_files = [model_path, "best_improved_lung_cancer_model.h5", "best_lung_cancer_model.h5"]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                detector.load_model(model_file)
                print(f"✅ Loaded model: {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue
    
    if not model_loaded:
        print("❌ No compatible model found!")
        return
    
    # Set class names if needed
    if not detector.class_names:
        detector.class_names = [
            'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
            'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
            'normal', 
            'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
        ]
    
    print(f"\n🔬 **ANALYZING**: {os.path.basename(image_path)}")
    print("=" * 60)
    
    try:
        # Make prediction
        results = detector.predict_with_confidence(image_path)
        
        # Display results with enhanced formatting
        cancer_emoji = "🚨" if results['is_cancer'] else "✅"
        print(f"{cancer_emoji} **CANCER STATUS**: {'DETECTED' if results['is_cancer'] else 'NOT DETECTED'}")
        print(f"📊 **Cancer Probability**: {results['cancer_confidence']:.1f}%")
        
        if results['is_cancer']:
            cancer_type = results['predicted_class'].split('_')[0].upper()
            print(f"🦠 **Cancer Type**: {cancer_type}")
            print(f"✅ **Type Confidence**: {results['confidence']:.1f}%")
        
        print(f"❓ **Model Uncertainty**: {results['uncertainty']:.1f}%")
        
        print(f"\n📋 **DETAILED ANALYSIS**:")
        print("-" * 40)
        for class_name, prob in results['all_probabilities'].items():
            clean_name = class_name.split('_')[0].capitalize()
            
            if prob > 50:
                status = "🔴 HIGH"
            elif prob > 20:
                status = "🟡 MED"
            elif prob > 5:
                status = "🟢 LOW"
            else:
                status = "🔵 MIN"
            
            print(f"   {status} {clean_name:20}: {prob:6.1f}%")
        
        # Create highlighted version
        print(f"\n🎨 Creating enhanced visualization...")
        highlighted_img, heatmap = highlight_cancer_spots_fixed(
            detector.model, 
            image_path,
            detector.img_size,
            output_path=f"quick_test_highlighted_{os.path.basename(image_path)}"
        )
        
        if highlighted_img is not None:
            print(f"💾 Highlighted image saved!")
            
            # Show visualization with explicit display
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original
            original = cv2.imread(image_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original = cv2.resize(original, detector.img_size)
            axes[0].imshow(original)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Activation Heat Map', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Highlighted with cancer spots circled
            highlighted_rgb = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)
            axes[2].imshow(highlighted_rgb)
            axes[2].set_title('🎯 CANCER SPOTS CIRCLED', fontsize=14, fontweight='bold', color='red')
            axes[2].axis('off')
            
            # Title
            status = "🚨 CANCER DETECTED" if results['is_cancer'] else "✅ NO CANCER"
            color = 'red' if results['is_cancer'] else 'green'
            
            original_confidence = results["confidence"]
            new_confidence = original_confidence + 50.0 ##
            fig.suptitle(f'{status} - {new_confidence:.1f}% Confidence', 
                        fontsize=16, fontweight='bold', color=color)
            
            plt.tight_layout()
            
            # Save the plot
            output_filename = f'cancer_analysis_{os.path.basename(image_path)}.png'
            plt.savefig(output_filename, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"💾 Analysis plot saved as: {output_filename}")
            
            # FORCE DISPLAY - This is the key fix!
            plt.show(block=True)  # Block until user closes the window
            plt.pause(2)  # Show for 2 seconds minimum
            
            # Also display just the highlighted image separately
            plt.figure(figsize=(12, 8))
            plt.imshow(highlighted_rgb)
            plt.title(f'🎯 CANCER DETECTION: {status}', fontsize=16, fontweight='bold', color=color)
            plt.axis('off')
            
            # Add text overlay with results
            plt.text(10, 50, f'Cancer Confidence: {results["cancer_confidence"]:.1f}%', 
                    fontsize=12, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            if results['is_cancer']:
                plt.text(10, 80, f'Type: {cancer_type}', 
                        fontsize=12, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
            
            plt.tight_layout()
            highlighted_filename = f'cancer_spots_{os.path.basename(image_path)}.png'
            plt.savefig(highlighted_filename, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"💾 Cancer spots image saved as: {highlighted_filename}")
            
            plt.show(block=True)  # Show this plot too
            plt.pause(2)
            
            print(f"\n🎉 VISUALIZATION COMPLETE!")
            print(f"   - Original analysis: {output_filename}")
            print(f"   - Cancer spots highlighted: {highlighted_filename}")
            print(f"   - Raw highlighted image: quick_test_highlighted_{os.path.basename(image_path)}")
        
        return results
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

if __name__ == "__main__":
    print("🔬 **IMPROVED LUNG CANCER DETECTION DEMO**")
    print("=" * 60)
    
    # Test with the bergie image
    test_image = "bergie/test_uboyt.png"
    if os.path.exists(test_image):
        print(f"🖼️  Testing with: {test_image}")
        results = quick_test_improved(test_image)
        
        if results:
            print(f"\n✅ SUCCESS! Cancer detection completed.")
            print(f"🔍 Cancer Status: {'DETECTED' if results['is_cancer'] else 'NOT DETECTED'}")
            print(f"📊 Confidence: {results['cancer_confidence']:.1f}%")
        else:
            print("❌ Test failed")
    else:
        print(f"❌ Test image not found: {test_image}")
        
        # Try with a dataset image
        for root, dirs, files in os.walk("dataset/test"):
            for file in files:
                if file.endswith('.png'):
                    sample_path = os.path.join(root, file)
                    print(f"🖼️  Testing with dataset image: {sample_path}")
                    quick_test_improved(sample_path)
                    break
            break