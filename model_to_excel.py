import pandas as pd
import numpy as np
import h5py
import os
from tensorflow.keras.models import load_model
import pickle

def model_to_excel(model_path, output_excel_path="lung_cancer_model_analysis.xlsx"):
    """
    Convert lung cancer model to Excel format with comprehensive analysis
    """
    print(f"🔬 Loading model from: {model_path}")
    
    try:
        # Load the model
        model = load_model(model_path)
        print(f"✅ Model loaded successfully!")
        
        # Create Excel writer
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            
            # 1. Model Summary Sheet
            print("📊 Creating Model Summary...")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            
            summary_df = pd.DataFrame({
                'Model Architecture': model_summary
            })
            summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
            
            # 2. Layer Details Sheet
            print("🔍 Analyzing layers...")
            layer_data = []
            total_params = 0
            
            for i, layer in enumerate(model.layers):
                layer_info = {
                    'Layer_Index': i,
                    'Layer_Name': layer.name,
                    'Layer_Type': type(layer).__name__,
                    'Input_Shape': str(layer.input_shape) if hasattr(layer, 'input_shape') else 'N/A',
                    'Output_Shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                    'Trainable_Params': layer.count_params() if hasattr(layer, 'count_params') else 0,
                    'Trainable': layer.trainable if hasattr(layer, 'trainable') else 'N/A'
                }
                
                # Add layer-specific information
                if hasattr(layer, 'activation'):
                    layer_info['Activation'] = str(layer.activation).split(' ')[1] if layer.activation else 'None'
                
                if hasattr(layer, 'filters'):
                    layer_info['Filters'] = layer.filters
                
                if hasattr(layer, 'kernel_size'):
                    layer_info['Kernel_Size'] = str(layer.kernel_size)
                
                if hasattr(layer, 'strides'):
                    layer_info['Strides'] = str(layer.strides)
                
                if hasattr(layer, 'units'):
                    layer_info['Units'] = layer.units
                
                if hasattr(layer, 'dropout'):
                    layer_info['Dropout_Rate'] = layer.dropout if hasattr(layer, 'dropout') else 'N/A'
                
                layer_data.append(layer_info)
                total_params += layer_info['Trainable_Params']
            
            layers_df = pd.DataFrame(layer_data)
            layers_df.to_excel(writer, sheet_name='Layer_Details', index=False)
            
            # 3. Model Configuration
            print("⚙️ Extracting configuration...")
            config_data = {
                'Property': [
                    'Model Name',
                    'Total Layers',
                    'Total Parameters',
                    'Trainable Parameters',
                    'Input Shape',
                    'Output Shape',
                    'Model Type',
                    'Framework'
                ],
                'Value': [
                    model.name,
                    len(model.layers),
                    model.count_params(),
                    sum([layer.count_params() for layer in model.layers if layer.trainable]),
                    str(model.input_shape),
                    str(model.output_shape),
                    type(model).__name__,
                    'TensorFlow/Keras'
                ]
            }
            
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Model_Config', index=False)
            
            # 4. Weight Statistics (for each layer with weights)
            print("📈 Calculating weight statistics...")
            weight_stats = []
            
            for layer in model.layers:
                if layer.get_weights():
                    weights = layer.get_weights()
                    for i, weight_matrix in enumerate(weights):
                        if len(weight_matrix.shape) > 0:  # Skip scalar weights
                            stats = {
                                'Layer_Name': layer.name,
                                'Weight_Index': i,
                                'Weight_Type': 'Kernel' if i == 0 else 'Bias' if i == 1 else f'Weight_{i}',
                                'Shape': str(weight_matrix.shape),
                                'Size': weight_matrix.size,
                                'Mean': float(np.mean(weight_matrix)),
                                'Std': float(np.std(weight_matrix)),
                                'Min': float(np.min(weight_matrix)),
                                'Max': float(np.max(weight_matrix)),
                                'Zero_Count': int(np.sum(weight_matrix == 0)),
                                'Non_Zero_Percentage': float((np.count_nonzero(weight_matrix) / weight_matrix.size) * 100)
                            }
                            weight_stats.append(stats)
            
            if weight_stats:
                weights_df = pd.DataFrame(weight_stats)
                weights_df.to_excel(writer, sheet_name='Weight_Statistics', index=False)
            
            # 5. Performance Metrics (if metadata exists)
            print("📊 Looking for performance data...")
            metadata_files = [
                'improved_lung_cancer_detector_metadata.pkl',
                'lung_cancer_model_metadata.pkl',
                'model_metadata.pkl'
            ]
            
            performance_data = []
            for metadata_file in metadata_files:
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'rb') as f:
                            metadata = pickle.load(f)
                            
                        if isinstance(metadata, dict):
                            for key, value in metadata.items():
                                performance_data.append({
                                    'Metric': key,
                                    'Value': str(value),
                                    'Source': metadata_file
                                })
                        break
                    except:
                        continue
            
            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                performance_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            
            # 6. Class Information
            print("🏷️ Adding class information...")
            class_info = [
                {
                    'Class_Index': 0,
                    'Class_Name': 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
                    'Description': 'Adenocarcinoma - Left Lower Lobe, Stage Ib',
                    'Cancer_Type': 'Adenocarcinoma'
                },
                {
                    'Class_Index': 1,
                    'Class_Name': 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
                    'Description': 'Large Cell Carcinoma - Left Hilum, Stage IIIa',
                    'Cancer_Type': 'Large Cell Carcinoma'
                },
                {
                    'Class_Index': 2,
                    'Class_Name': 'normal',
                    'Description': 'Normal lung tissue - No cancer detected',
                    'Cancer_Type': 'Normal'
                },
                {
                    'Class_Index': 3,
                    'Class_Name': 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa',
                    'Description': 'Squamous Cell Carcinoma - Left Hilum, Stage IIIa',
                    'Cancer_Type': 'Squamous Cell Carcinoma'
                }
            ]
            
            classes_df = pd.DataFrame(class_info)
            classes_df.to_excel(writer, sheet_name='Class_Information', index=False)
            
            # 7. Model Usage Instructions
            print("📝 Adding usage instructions...")
            instructions = [
                {
                    'Step': 1,
                    'Action': 'Load Model',
                    'Code': 'model = load_model("best_fine_tuned_lung_cancer_model.h5")',
                    'Description': 'Load the trained model using TensorFlow/Keras'
                },
                {
                    'Step': 2,
                    'Action': 'Prepare Image',
                    'Code': 'img = load_img(image_path, target_size=(384, 384))',
                    'Description': 'Load and resize image to model input size'
                },
                {
                    'Step': 3,
                    'Action': 'Preprocess',
                    'Code': 'img_array = img_to_array(img) / 255.0',
                    'Description': 'Convert to array and normalize pixel values'
                },
                {
                    'Step': 4,
                    'Action': 'Predict',
                    'Code': 'prediction = model.predict(np.expand_dims(img_array, axis=0))',
                    'Description': 'Make prediction on the preprocessed image'
                },
                {
                    'Step': 5,
                    'Action': 'Interpret',
                    'Code': 'class_idx = np.argmax(prediction)',
                    'Description': 'Get the predicted class index'
                }
            ]
            
            instructions_df = pd.DataFrame(instructions)
            instructions_df.to_excel(writer, sheet_name='Usage_Instructions', index=False)
            
        print(f"✅ Excel file created successfully: {output_excel_path}")
        print(f"📊 Sheets created:")
        print(f"   - Model_Summary: Overall model architecture")
        print(f"   - Layer_Details: Detailed layer information")
        print(f"   - Model_Config: Model configuration parameters")
        print(f"   - Weight_Statistics: Statistical analysis of model weights")
        print(f"   - Class_Information: Cancer class definitions")
        print(f"   - Usage_Instructions: How to use the model")
        
        if performance_data:
            print(f"   - Performance_Metrics: Model performance data")
        
        return output_excel_path
        
    except Exception as e:
        print(f"❌ Error converting model to Excel: {e}")
        return None

if __name__ == "__main__":
    print("🔬 **LUNG CANCER MODEL TO EXCEL CONVERTER**")
    print("=" * 60)
    
    # Find the best model
    model_files = [
        "best_fine_tuned_lung_cancer_model.h5",
        "best_improved_lung_cancer_model.h5", 
        "best_lung_cancer_model.h5",
        "improved_lung_cancer_detector.h5",
        "lung_cancer_detector.h5"
    ]
    
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"📂 Found model: {model_file}")
            
            output_file = f"lung_cancer_model_analysis_{os.path.splitext(model_file)[0]}.xlsx"
            result = model_to_excel(model_file, output_file)
            
            if result:
                print(f"🎉 SUCCESS! Model converted to Excel format.")
                print(f"📁 Excel file saved as: {result}")
                model_found = True
                break
            else:
                print(f"❌ Failed to convert {model_file}")
    
    if not model_found:
        print("❌ No compatible model files found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")