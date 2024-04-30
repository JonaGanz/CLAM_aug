import numpy as np
import torch

from numpy.random import uniform
from PIL import Image
from torchvision import transforms
from torchstain.torch.normalizers import TorchMacenkoNormalizer
from torchstain.numpy.normalizers import NumpyMacenkoNormalizer
from numpy.linalg import LinAlgError


class NumpyMarcenkoAugmentor(NumpyMacenkoNormalizer):
    def __init__(self,sigma1 = 0.2, sigma2 = 0.2):
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
    def compute_matrices(self, I, Io, alpha, beta):
        return self._NumpyMacenkoNormalizer__compute_matrices(I, Io, alpha, beta)
        
    def augment(self, input, Io:int = 240, alpha=1, beta = 0.15):
        # convert PIL to numpy image
        I = self.convert_pil_to_numpy(input)
        try:
            h, w, c = I.shape
            I = I.reshape((-1,3))
            # get stain matrix and concentration
            HE, C, _ = self.compute_matrices(I, Io, alpha, beta)
            
            n_stains = C.shape[0]
            for i in range(n_stains):
                alpha = uniform(1 - self.sigma1, 1 + self.sigma1)
                beta = uniform(-self.sigma2, self.sigma2)
                C[i,:] *= alpha
                C[i,:] += beta
            
            # recreate the image
            I = np.multiply(Io, np.exp(-HE.dot(C)))
            I[I > 255] = 255
            I = np.reshape(I.T, (h, w, c)).astype(np.uint8)
            
            I = self.convert_numpy_to_pil(I)
        # this error occurs with "empty" patches which are mostly
        # white
        except LinAlgError:
            return input
        return I
    
    def convert_pil_to_numpy(self,img):
        return np.array(img)
    
    def convert_numpy_to_pil(self,img):
        return Image.fromarray(img)

class TorchMarcenkoAugmentor(TorchMacenkoNormalizer):
    def __init__(self,sigma1 = 0.2, sigma2 = 0.2):
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        self.pil_to_tensor = transforms.Compose([transforms.PILToTensor()])
        self.tensor_to_pil = transforms.Compose([transforms.ToPILImage()])
        
    def compute_matrices(self, I, Io, alpha, beta):
        return self._TorchMacenkoNormalizer__compute_matrices(I, Io, alpha, beta)
        
    def augment(self, I, Io:int = 240, alpha=1, beta = 0.15):
        # convert PIL to numpy image
        I = self.pil_to_tensor(I)
        c, h, w = I.shape
        # get stain matrix and concentration
        try:
            HE, C, _ = self.compute_matrices(I, Io, alpha, beta)
            n_stains = C.shape[0]
            for i in range(n_stains):
                alpha = uniform(1 - self.sigma1, 1 + self.sigma1)
                beta = uniform(-self.sigma2, self.sigma2)
                C[i,:] *= alpha
                C[i,:] += beta
            
            # recreate the image
            I = Io * torch.exp(-torch.matmul(HE,C))
            I[I > 255] = 255
            I = I.T.reshape(h,w,c).to(torch.uint8).transpose(2,0)
            
            I = self.tensor_to_pil(I)
        # this error occurs with "empty" patches which are mostly
        # white
        except:
            I = self.tensor_to_pil(I)
        
        return I
    
class MarcenkoAugmentation(object):
    """Apply Marcenko stain augmentation"""
    def __init__(self,sigma1=0.2, sigma2=0.2, backend:str = 'torch') -> None:
        assert backend in ['torch', 'numpy'], 'unkown backend, select numpy or torch'
        if backend == 'torch':
            self.augmentor = TorchMarcenkoAugmentor(sigma1=sigma1, sigma2=sigma2)
        elif backend == 'numpy':
            self.augmentor = NumpyMarcenkoAugmentor(sigma1=sigma1, sigma2=sigma2)
    
    def __call__(self,img):
        return self.augmentor.augment(img)