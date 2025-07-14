# evaluate/evaluate_models.py
import matplotlib.pyplot as plt

def plot_comparison(gray, unet_out, gan_out, ground_truth):
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(gray.squeeze(0).cpu(), cmap='gray')
    axs[0].set_title('Grayscale')
    axs[1].imshow(unet_out.permute(1, 2, 0).cpu().clamp(0, 1))
    axs[1].set_title('U-Net Output')
    axs[2].imshow(gan_out.permute(1, 2, 0).cpu().clamp(0, 1))
    axs[2].set_title('GAN Output')
    axs[3].imshow(ground_truth.permute(1, 2, 0).cpu())
    axs[3].set_title('Ground Truth')
    for ax in axs:
        ax.axis('off')
    plt.show()
