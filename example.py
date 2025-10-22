"""
ConvRec Face Recognition - Quick Example
Shows how to use the model for face verification
"""

from face_recognition import FaceRecognition

# Initialize the ConvRec model
print("Loading ConvRec model...")
model = FaceRecognition('best_model.pth')

# Example 1: Verify if two faces are the same person
print("\n--- Face Verification Example ---")
# Replace with your image paths
image1 = 'data/img1.jpg'
image2 = 'data/img2.jpg'

try:
    result = model.verify_faces(image1, image2, threshold=0.5)

    if result['is_same']:
        print(f"✓ These are the SAME person")
    else:
        print(f"✗ These are DIFFERENT people")

    print(f"Similarity score: {result['similarity']:.3f}")
    print(f"Threshold used: {result['threshold']}")

except Exception as e:
    print(f"Please ensure you have images in the data folder")
    print(f"Error: {e}")

# Example 2: Build a gallery and search
print("\n--- Gallery Search Example ---")
try:
    # Build gallery from data folder
    print("Building face gallery from data folder...")
    gallery = model.build_gallery('data/', max_per_person=3)

    if gallery:
        # Search for a face
        query_image = 'data/img1.jpg'
        print(f"\nSearching for: {query_image}")

        matches = model.search_in_gallery(query_image, gallery, top_k=3)

        print("\nTop matches:")
        for i, (person, similarity) in enumerate(matches, 1):
            print(f"{i}. {person}: {similarity:.3f}")
    else:
        print("No gallery could be built. Add folders with images to data/")

except Exception as e:
    print(f"Error in gallery search: {e}")

print("\n--- ConvRec Model Ready ---")
print("Model successfully loaded and ready for face recognition tasks!")
print("For more examples, check the documentation at:")
print("https://huggingface.co/convaiinnovations/convrec-face-recognition")