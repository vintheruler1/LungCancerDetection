from PIL import Image
import pandas as pd
import os
from pathlib import Path
import concurrent.futures
from threading import Lock
import time

# Thread lock for thread-safe operations
print_lock = Lock()

def thread_safe_print(message):
    """Thread-safe printing"""
    with print_lock:
        print(message)

def process_image_to_chunks(image_path):
    """Process an image and return RGB data for each individual pixel"""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Resize to half the original size
        new_width = width // 2
        new_height = height // 2
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        width, height = img.size
        
        thread_safe_print(f"Processing image resized to: {width}x{height} - {os.path.basename(image_path)}")
        
        # Get all pixel data
        pixels = list(img.getdata())
        
        # Create column headers with spaces and blank columns: C1 R, C1 G, C1 B, (blank), C2 R, C2 G, C2 B, (blank), etc.
        columns = []
        for col in range(width):
            columns.extend([f"C{col+1} R", f"C{col+1} G", f"C{col+1} B"])
            # Add blank column after each RGB set (except the last one)
            if col < width - 1:
                columns.append("")
        
        # Create data structure
        data = []
        
        # Process each row in the image
        for row in range(height):
            row_data = []
            
            # For each column in this row
            for col in range(width):
                pixel_index = row * width + col
                pixel = pixels[pixel_index]
                
                # Add R, G, B values for this pixel
                row_data.extend([pixel[0], pixel[1], pixel[2]])
                # Add blank cell after each RGB set (except the last one)
                if col < width - 1:
                    row_data.append("")
            
            data.append(row_data)
        
        # Create DataFrame with proper row indices (R1, R2, etc.)
        df = pd.DataFrame(data, columns=columns)
        df.index = [f"R{i+1}" for i in range(height)]
        
        return image_path, df
    except Exception as e:
        thread_safe_print(f"Error processing {image_path}: {e}")
        return image_path, None

def process_images_parallel(image_paths, max_workers=4):
    """Process multiple images in parallel"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_image_to_chunks, path): path for path in image_paths}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path, df = future.result()
            results[path] = df
    
    return results

def process_all_training_images():
    """Process all training images and create separate Excel files for each cancer type"""
    train_path = "dataset/train"
    
    if not os.path.exists(train_path):
        print(f"Path not found: {train_path}")
        return
    
    # Get all category folders in train directory
    categories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    print(f"Found categories: {categories}")
    
    for category in categories:
        category_path = os.path.join(train_path, category)
        
        # Create a clean name for the Excel file
        clean_category_name = category.split('_')[0]  # Get just the cancer type name
        excel_filename = f"{clean_category_name}_training_pixel_data.xlsx"
        
        print(f"\nProcessing category: {category}")
        print(f"Excel file will be: {excel_filename}")
        
        # Get all PNG files in this category folder
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith('.png')]
        
        if not image_files:
            print(f"No PNG files found in {category_path}")
            continue
        
        print(f"Found {len(image_files)} images to process")
        
        # Create full paths for all images
        image_paths = [os.path.join(category_path, f) for f in image_files]
        
        try:
            # Process images in parallel
            start_time = time.time()
            print(f"Starting parallel processing with 4 threads...")
            
            results = process_images_parallel(image_paths, max_workers=4)
            
            processing_time = time.time() - start_time
            print(f"Parallel processing completed in {processing_time:.2f} seconds")
            
            # Create Excel file
            print("Creating Excel file...")
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                processed_count = 0
                
                for image_path, df in results.items():
                    if df is not None:
                        # Get just the filename for sheet name
                        image_file = os.path.basename(image_path)
                        sheet_name = os.path.splitext(image_file)[0]
                        
                        # Remove problematic characters and limit length
                        sheet_name = sheet_name.replace('(', '').replace(')', '').replace(' ', '_').replace('.', '_')
                        if len(sheet_name) > 25:  # Leave room for potential suffixes
                            sheet_name = sheet_name[:25]
                        
                        # Ensure unique sheet name
                        original_name = sheet_name
                        counter = 1
                        while sheet_name in [ws.title for ws in writer.book.worksheets if writer.book.worksheets]:
                            sheet_name = f"{original_name}_{counter}"
                            counter += 1
                        
                        # Write to Excel sheet
                        df.to_excel(writer, sheet_name=sheet_name, index=True)
                        processed_count += 1
                        print(f"Added sheet: {sheet_name}")
                
                print(f"Completed category {category}: {processed_count} images processed")
                print(f"Excel file saved as: {excel_filename}")
                
        except Exception as e:
            print(f"Error creating Excel file for {category}: {e}")

def main():
    """Main function to process all training images"""
    print("Processing all training lung cancer images with multithreading...")
    print("This will create separate Excel files for each cancer type.")
    
    start_time = time.time()
    process_all_training_images()
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()