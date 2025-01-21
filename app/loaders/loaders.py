from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from app.core.image_models import AsyncVisionLanguageModelClassifier, MoondreamProcessor


class ImageLoader:
    """
    1. Loads and preprocesses images from a folder to memory.
    2. Prepares batches of images for model input.
    """
    def __init__(self, folder_path, target_size=(512, 512), max_workers=4):
        self.folder_path = folder_path
        self.target_size = target_size
        self.max_workers = max_workers
        self.image_data = []  # Store loaded images

        # Auto-load images on initialization
        self.load_images()

    def _load_and_preprocess_image(self, image_path):
        """Loads an image from disk and preprocesses it (resize)."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.target_size, Image.LANCZOS)
                return os.path.basename(image_path), img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def load_images(self):
        """Loads and preprocesses all images in parallel, storing them in self.image_data."""
        image_files = [
            os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._load_and_preprocess_image, image_files)

        self.image_data = [result for result in results if result]  # Store as [(filename, PIL.Image)]
        print(f"Loaded {len(self.image_data)} images from {self.folder_path}")

    def batch_images(self, batch_size=8):
        """Generates batches of images while keeping filenames linked."""
        for i in range(0, len(self.image_data), batch_size):
            batch = self.image_data[i : i + batch_size]
            filenames, images = zip(*batch)  # Separate filenames and images
            yield filenames, images


class ModelLoader:
    def __init__(self, model_type: str = "local"):
        """
        Initialize the image processor with specified model type.
        
        Args:
            model_type: "local" or "hosted" 
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        if self.model_type == "local":
            return MoondreamProcessor()
        elif self.model_type == "hosted":
            return AsyncVisionLanguageModelClassifier()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")