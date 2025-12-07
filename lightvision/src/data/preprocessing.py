from torchvision import transforms
from PIL import Image

def get_train_transform(img_size=224):
    """Training data augmentation and preprocessing"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform(img_size=224):
    """Validation / test preprocessing"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def resize_image(image: Image.Image, size=(224,224)) -> Image.Image:
    """Helper function to resize PIL images"""
    return image.resize(size)