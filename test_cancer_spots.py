import os
from demo_cancer_detection import quick_test_improved

print("🔬 Testing Cancer Detection with Spot Highlighting")
print("=" * 60)

# Test with a sample image
test_image = "bergie/test_uboyt.png"

if os.path.exists(test_image):
    print(f"Testing image: {test_image}")
    print("Running cancer detection...")
    
    # Run the cancer detection
    results = quick_test_improved(test_image)
    
    if results:
        print("\n✅ Cancer detection completed!")
        print("🔍 Look for these files in your directory:")
        print("   - highlighted images with cancer spots circled")
        print("   - analysis images with side-by-side comparisons")
        
        # List generated files
        generated_files = []
        for file in os.listdir("."):
            if file.startswith(("highlighted_", "quick_test_", "analysis_")):
                generated_files.append(file)
        
        if generated_files:
            print(f"\n📁 Generated {len(generated_files)} files:")
            for file in generated_files:
                print(f"   - {file}")
        else:
            print("\n⚠️  No visualization files found yet")
    else:
        print("❌ Cancer detection failed")
else:
    print(f"❌ Test image not found: {test_image}")
    
    # Try to find any PNG files in dataset/test
    print("\nLooking for available test images...")
    for root, dirs, files in os.walk("dataset/test"):
        for file in files[:3]:  # Show first 3 files
            if file.endswith('.png'):
                full_path = os.path.join(root, file)
                print(f"Found: {full_path}")
                break

print("\nDone! Check your file directory for highlighted cancer detection images.")