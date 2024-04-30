from torchvision import transforms
from .stain_augmentation import MarcenkoAugmentation

def get_eval_transforms(mean, std, target_img_size = -1, stain_augmentation = False):
    trsforms = []
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    if stain_augmentation:
        trsforms.append(MarcenkoAugmentation(sigma1 = 0.2, sigma2 = 0.2, backend='torch'))
    trsforms.append(transforms.ToTensor())
    trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)
    return trsforms