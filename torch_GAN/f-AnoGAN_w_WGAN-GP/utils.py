
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    # H = 64
    # W = 64
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

def compare_images(real_img, generated_img, i, score, reverse=False, threshold=50):
    real_img = np.transpose(real_img.cpu().data.numpy().squeeze(), (1, 2, 0))
    dim_1 = np.shape(real_img)[0]
    dim_2 = np.shape(real_img)[1]
    channels = np.shape(real_img)[2]
    real_img = real_img.reshape(dim_1, dim_2, channels)
    generated_img = np.transpose(generated_img.cpu().data.numpy().squeeze(), (1, 2, 0))
    generated_img = generated_img.reshape(dim_1, dim_2, channels)

    negative = np.zeros_like(real_img)

    if not reverse:
        diff_img = real_img - generated_img
    else:
        diff_img = generated_img - real_img

    diff_img[diff_img <= threshold] = 0

    anomaly_img = [np.zeros(shape=(dim_1, dim_2, channels)), np.zeros(shape=(dim_1, dim_2, channels)), np.zeros(shape=(dim_1, dim_2, channels))]
    anomaly_img[0] = (real_img - diff_img) * 255
    anomaly_img[1] = (real_img - diff_img) * 255
    anomaly_img[2] = (real_img - diff_img) * 255
    anomaly_img[0] = anomaly_img[0] + diff_img

    anomaly_img = [anomaly_img[0].astype(np.uint8), anomaly_img[1].astype(np.uint8), anomaly_img[2].astype(np.uint8)]

    fig, plots = plt.subplots(1, 4)

    fig.suptitle(f'Anomaly - (anomaly score: {score:.4})')

    fig.set_figwidth(20)
    fig.set_tight_layout(True)
    plots = plots.reshape(-1)
    plots[0].imshow(real_img, cmap='bone', label='real')
    plots[1].imshow(generated_img, cmap='bone')
    plots[2].imshow(diff_img, cmap='bone')
    plots[3].imshow(anomaly_img[0], cmap='bone')


    plots[0].set_title('real')
    plots[1].set_title('generated')
    plots[2].set_title('difference')
    plots[3].set_title('Anomaly Detection')
    plt.show()