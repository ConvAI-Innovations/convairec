"""
Face Deduplication Module
Find and manage duplicate faces in image collections
"""

import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from face_recognition import FaceRecognition
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDeduplication:
    """Face deduplication using similarity matching"""

    def __init__(self, model_path='best_model.pth'):
        """Initialize with face recognition model"""
        self.face_rec = FaceRecognition(model_path)

    def find_duplicates(self, folder_path, threshold=0.5):
        """
        Find duplicate faces in a folder

        Args:
            folder_path: Path to folder containing images
            threshold: Similarity threshold for considering duplicates

        Returns:
            List of duplicate groups
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []

        folder_path = Path(folder_path)
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'**/*{ext}'))
            image_files.extend(folder_path.glob(f'**/*{ext.upper()}'))

        image_files = list(set(image_files))
        logger.info(f"Found {len(image_files)} images")

        if len(image_files) < 2:
            return []

        # Extract embeddings
        embeddings = []
        valid_files = []

        for img_path in image_files:
            try:
                emb = self.face_rec.extract_embedding(str(img_path))
                if emb is not None:
                    embeddings.append(emb)
                    valid_files.append(str(img_path))
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")

        if len(embeddings) < 2:
            return []

        # Compute similarity matrix
        embeddings = np.array(embeddings)
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Find duplicates
        duplicate_groups = []
        processed = set()

        for i in range(len(valid_files)):
            if i in processed:
                continue

            # Find all images similar to image i
            similar_indices = np.where(similarity_matrix[i] > threshold)[0]

            if len(similar_indices) > 1:
                group = [valid_files[idx] for idx in similar_indices]
                duplicate_groups.append(group)
                processed.update(similar_indices)

        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups

    def remove_duplicates(self, folder_path, threshold=0.5, keep='first', dry_run=True):
        """
        Remove duplicate faces from folder

        Args:
            folder_path: Path to folder containing images
            threshold: Similarity threshold
            keep: Strategy ('first', 'last', 'best')
            dry_run: If True, only show what would be removed

        Returns:
            Dictionary with kept and removed files
        """
        duplicate_groups = self.find_duplicates(folder_path, threshold)

        kept_files = []
        removed_files = []

        for group in duplicate_groups:
            if keep == 'first':
                keep_file = group[0]
                remove_files = group[1:]
            elif keep == 'last':
                keep_file = group[-1]
                remove_files = group[:-1]
            else:  # 'best' - keep largest file
                sizes = [(f, os.path.getsize(f)) for f in group]
                sizes.sort(key=lambda x: x[1], reverse=True)
                keep_file = sizes[0][0]
                remove_files = [f for f in group if f != keep_file]

            kept_files.append(keep_file)
            removed_files.extend(remove_files)

        if not dry_run:
            for file_path in removed_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.error(f"Could not remove {file_path}: {e}")
        else:
            logger.info(f"DRY RUN - Would remove {len(removed_files)} files")

        return {
            'kept': kept_files,
            'removed': removed_files,
            'dry_run': dry_run
        }

# Example usage
if __name__ == '__main__':
    dedup = FaceDeduplication('best_model.pth')

    # Find duplicates
    if os.path.exists('data'):
        duplicates = dedup.find_duplicates('data', threshold=0.5)
        for i, group in enumerate(duplicates, 1):
            print(f"\nDuplicate group {i}:")
            for img in group:
                print(f"  - {img}")

    print("Face deduplication ready!")