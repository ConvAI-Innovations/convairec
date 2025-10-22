"""
ConvRec Face Recognition Inference Module
High-performance face recognition using ConvRec proprietary model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceModel(nn.Module):
    """Face recognition model with ResNet50 backbone"""
    def __init__(self, embedding_size=512):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)
        return F.normalize(x, p=2, dim=1)

class FaceRecognition:
    """Main class for face recognition tasks"""

    def __init__(self, model_path='best_model.pth', device=None):
        """
        Initialize face recognition model

        Args:
            model_path: Path to model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

        logger.info(f"Model loaded on {self.device}")

    def _load_model(self, model_path):
        """Load the trained model"""
        model = FaceModel(embedding_size=512).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)

            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")

        model.eval()
        return model

    def _get_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_embedding(self, image_path):
        """
        Extract face embedding from an image

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            512-dimensional embedding vector
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding.cpu().numpy().squeeze()

    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        # Embeddings are already normalized
        return np.dot(embedding1, embedding2)

    def verify_faces(self, image1_path, image2_path, threshold=0.5):
        """
        Verify if two faces belong to the same person

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            threshold: Similarity threshold for verification

        Returns:
            Dictionary with verification results
        """
        emb1 = self.extract_embedding(image1_path)
        emb2 = self.extract_embedding(image2_path)

        similarity = self.compute_similarity(emb1, emb2)
        is_same = similarity > threshold

        return {
            'is_same': bool(is_same),
            'similarity': float(similarity),
            'threshold': threshold
        }

    def build_gallery(self, folder_path, max_per_person=5):
        """
        Build a gallery of known faces from a folder

        Args:
            folder_path: Path to folder with subfolders for each person
            max_per_person: Maximum images per person to use

        Returns:
            Gallery dictionary with embeddings
        """
        gallery = {}
        folder_path = Path(folder_path)

        for person_dir in folder_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                embeddings = []

                image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))

                for img_path in image_files[:max_per_person]:
                    emb = self.extract_embedding(str(img_path))
                    if emb is not None:
                        embeddings.append(emb)

                if embeddings:
                    # Average embeddings for robustness
                    avg_emb = np.mean(embeddings, axis=0)
                    avg_emb = avg_emb / np.linalg.norm(avg_emb)
                    gallery[person_name] = avg_emb
                    logger.info(f"Added {person_name} to gallery ({len(embeddings)} images)")

        logger.info(f"Gallery built with {len(gallery)} persons")
        return gallery

    def search_in_gallery(self, query_image_path, gallery, top_k=5):
        """
        Search for a face in the gallery

        Args:
            query_image_path: Path to query image
            gallery: Gallery dictionary from build_gallery
            top_k: Number of top matches to return

        Returns:
            List of (person_name, similarity) tuples
        """
        query_emb = self.extract_embedding(query_image_path)

        similarities = {}
        for person_name, gallery_emb in gallery.items():
            sim = self.compute_similarity(query_emb, gallery_emb)
            similarities[person_name] = sim

        # Sort by similarity
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_matches[:top_k]

    def are_same_person(self, image1_path, image2_path, threshold=0.5):
        """
        Simple boolean check if two images are the same person

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            threshold: Similarity threshold

        Returns:
            Boolean indicating if same person
        """
        result = self.verify_faces(image1_path, image2_path, threshold)
        return result['is_same']

# Example usage
if __name__ == '__main__':
    # Initialize model
    fr = FaceRecognition('best_model.pth')

    # Example: Verify two faces
    if os.path.exists('data'):
        # Find some test images
        test_images = list(Path('data').glob('**/*.jpg'))
        if len(test_images) >= 2:
            result = fr.verify_faces(str(test_images[0]), str(test_images[1]))
            print(f"Verification result: {result}")

    print("Face recognition model ready for inference!")