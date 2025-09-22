# Import modules
import numpy as np
import matplotlib.pyplot as plt
import cv2


# TODO: Adapt for RGB images. So far, only grayscale are supported.
def generate_error_map(
        img1: np.ndarray,
        img2: np.ndarray,
        do_visual: bool = True,
        do_save: bool = True
) -> None:
    """
    This function takes two images in input and compare them. At last it produces a
    2D heatmap to show major differences.

    Parameters:
    ----------
    img1, img2: np.ndarray[uint8 or float]
        input images to be compared.
    do_visual: bool
        it is a flag for plot visualization or not.
    do_save:
        it is a flag to store the plot or not.
    """

    # Define a lambda function according to the criterion
    comp_func = lambda img_true, img_est: np.abs(img_est - img_true)
    
    # Generate the error map
    error_map = comp_func(img1/255., img2/255.) * 8

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(error_map, cmap='hot', vmin=0, vmax=8)
    plt.colorbar()
    plt.title('Error Map')
    plt.axis('off')

    # Save plot
    if do_save:
        plt.savefig(f".png", dpi=300, bbox_inches='tight')
    
    # Visualize plot
    if do_visual:
        plt.show()

    plt.close


if __name__ == "__main__":
    """
    Test the function.
    """

    # Select images
    path1 = "geometry_shadow_decoupling/arch_0/results/folder_2025-09-16_17-31-20/epoch_14/depth/72.png"
    path2 = "/home/vision/Desktop/Datasets/CloudSat_NeRF_Datasets/colour_256_XY_12_3d0_training/072_d.png"

    # Load images
    img1 = cv2.imread(path1, 0) 
    img2 = cv2.imread(path2, 0)

    # Compare images
    generate_error_map(
        img1=img1,
        img2=img2,
        do_visual=True,
        do_save=True
    )
