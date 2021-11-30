import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte, img_as_float32

class Blend():
    """
    Implements blending methods
    """
    def __init__(self, method='pyramid', n_levels=4, n_levels_copy=0, device='cuda:0'):
        self.method = method
        if self.method not in ['pyramid', 'copy_paste', 'poisson']:
            raise NotImplementedError
        if self.method == 'pyramid' or self.method == 'poisson':
            self.n_levels = n_levels
            self.n_levels_copy = n_levels_copy
            self.device = device

    def paste_blending(self, imgA, imgB, mask):
        """
        Simple copy-paste blending
        """
        return imgA*(1.0-mask) + imgB*mask

    def poisson_blending(self, imgA, imgB, mask):
        """
        Poisson image editing
        """
        x,y,w,h = cv2.boundingRect(img_as_ubyte(mask[:,:,0]))
        center = (int(x+w*0.5),int(y+h*0.5))
        mixed = cv2.seamlessClone(img_as_ubyte(np.clip(imgB,0,1)), img_as_ubyte(np.clip(imgA,0,1)), img_as_ubyte(mask), center, cv2.NORMAL_CLONE)
        return img_as_float32(mixed)

    def gaussian_pyramid(self, img, num_levels):
        """
        Returns gaussian pyramid
        """
        lower = img.copy()
        gaussian_pyr = [lower]
        for i in range(num_levels):
            lower = cv2.pyrDown(lower)
            gaussian_pyr.append(lower)
        return gaussian_pyr

    def laplacian_pyramid(self, gaussian_pyr):
        """
        Returns laplacian pyramid
        """
        laplacian_top = gaussian_pyr[-1]
        num_levels = len(gaussian_pyr) - 1

        laplacian_pyr = [laplacian_top]
        for i in range(num_levels,0,-1):
            size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
            gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyr[i-1], gaussian_expanded)
            laplacian_pyr.append(laplacian)
        return laplacian_pyr

    def blend_laplacians(self, laplacian_A, laplacian_B, mask_pyr):
        """
        Blends the laplacian pyramids of the 2 images based on the mask pyramid
        """
        LS = []
        for i, (la,lb,mask) in enumerate(zip(laplacian_A,laplacian_B,mask_pyr)):
            if i<self.n_levels_copy:
                ls = la.copy()
            else:
                ls = lb * mask + la * (1.0 - mask)
            LS.append(ls)
        return LS

    def reconstruct(self, laplacian_pyr):
        """
        Reconstructs the original image from the laplacian pyramid
        """
        laplacian_top = laplacian_pyr[0]
        num_levels = len(laplacian_pyr) - 1
        for i in range(num_levels):
            size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
            laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
            laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        return laplacian_top

    def __call__(self, imgA, imgB, mask):
        """
        Main function for blending
            imgA, imgB, mask:   3-channel float32 images

        """
        # Resize if needed
        if imgA.shape!=imgB.shape:
            imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]), interpolation = cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (imgA.shape[1], imgA.shape[0]), interpolation = cv2.INTER_NEAREST)


        if self.method == 'copy_paste':
            new_img = self.paste_blending(imgA, imgB, mask)
        elif self.method == 'pyramid':
            # erode mask
            kernel_size = int(33*(imgA.shape[0]/256))
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoother = SoftErosion(kernel_size=kernel_size, threshold=0.8).to(self.device)
            mask = torch.tensor(mask).permute(2,0,1)[0:1].to(self.device)   # (1,H,W)
            _, mask = smoother(torch.unsqueeze(mask, 0))    # (1,1,H,W)
            mask = np.float32(mask.squeeze(0).permute(1,2,0).repeat(1,1,3).cpu().numpy())  # (H,W,3)

            # For image-A, calculate Gaussian and Laplacian
            gaussian_pyr_A = self.gaussian_pyramid(imgA, self.n_levels)
            laplacian_pyr_A = self.laplacian_pyramid(gaussian_pyr_A)
            # For image-B, calculate Gaussian and Laplacian
            gaussian_pyr_B = self.gaussian_pyramid(imgB, self.n_levels)
            laplacian_pyr_B = self.laplacian_pyramid(gaussian_pyr_B)
            # Calculate the Gaussian pyramid for the mask image and reverse it.
            mask_pyr = self.gaussian_pyramid(mask, self.n_levels)
            mask_pyr.reverse()
            # Blend the laplacians
            add_laplace = self.blend_laplacians(laplacian_pyr_A,laplacian_pyr_B,mask_pyr)
            # Reconstruct the images
            new_img = self.reconstruct(add_laplace)

        elif self.method == 'poisson':
            new_img = self.poisson_blending(imgA, imgB, mask)
            self.method = 'pyramid'
            new_img = self.__call__(imgA, new_img, mask)
            self.method = 'poisson'

            new_img = blursharpen(new_img)*mask+(1.0-mask)*imgA

        return new_img

def blursharpen (img, sharpen_mode=2, kernel_size=3, amount=3):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if amount > 0:
        if sharpen_mode == 1: #box
            kernel = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
            kernel[ kernel_size//2, kernel_size//2] = 1.0
            box_filter = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
            kernel = kernel + (kernel - box_filter) * amount
            return cv2.filter2D(img, -1, kernel)
        elif sharpen_mode == 2: #gaussian
            blur = cv2.GaussianBlur(img, (kernel_size, kernel_size) , 0)
            img = cv2.addWeighted(img, 1.0 + (0.5 * amount), blur, -(0.5 * amount), 0)
            return img
    elif amount < 0:
        n = -amount
        while n > 0:

            img_blur = cv2.medianBlur(img, 5)
            if int(n / 10) != 0:
                img = img_blur
            else:
                pass_power = (n % 10) / 10.0
                img = img*(1.0-pass_power)+img_blur*pass_power
            n = max(n-10,0)

        return img
    return img

class SoftErosion(torch.nn.Module):
    """ Applies *soft erosion* on a binary mask, that is similar to the
    `erosion morphology operation <https://en.wikipedia.org/wiki/Erosion_(morphology)>`_,
    returning both a soft mask and a hard binary mask.
    All values greater or equal to the the specified threshold will be set to 1 in both the soft and hard masks,
    the other values will be 0 in the hard mask and will be gradually reduced to 0 in the soft mask.
    Args:
        kernel_size (int): The size of the erosion kernel size
        threshold (float): The erosion threshold
        iterations (int) The number of times to apply the erosion kernel
    """
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        """ Apply the soft erosion operation.
        Args:
            x (torch.Tensor): A binary mask of shape (1, H, W)
        Returns:
            (torch.Tensor, torch.Tensor): Tuple containing:
                - soft_mask (torch.Tensor): The soft mask of shape (1, H, W)
                - hard_mask (torch.Tensor): The hard mask of shape (1, H, W)
        """
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask
