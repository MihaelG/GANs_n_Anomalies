"""
Training of WGAN-GP
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint, compare_images
from model import Discriminator, Generator, Encoder, initialize_weights

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
IMAGE_SIZE = 256
CHANNELS_IMG = 3
Z_DIM = 8
NUM_EPOCHS = 500
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

enc_epochs = 200
kappa = 1.0

train_gen_and_critic = False
train_enc = False

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# comment mnist above and uncomment below for training on CelebA
# dataset = datasets.ImageFolder(root="/home/mihael/ML/GANs_n_Anomalies/torch_GAN/dataset/celeb_dataset", transform=transforms)
dataset = datasets.ImageFolder(root="/home/mihael/ML/GANs_n_Anomalies/torch_GAN/dataset/Whole_Glomeruli_256", transform=transforms)
test_dataset = datasets.ImageFolder(root="/home/mihael/ML/GANs_n_Anomalies/torch_GAN/dataset/Sick_Glomeruli_256", transform=transforms)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])

if train_gen_and_critic:
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(4, Z_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs_kernel_3/real")
    writer_fake = SummaryWriter(f"logs_kernel_3/fake")
    # writer_enc = SummaryWriter(f"logs_kernel_3/enc")
    step = 0

    gen.train()
    critic.train()


    ### TRAINING OF GENERATOR AND CRITIC
    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 2 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1

    torch.save(critic.state_dict(), '/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netD_%d.pth' % NUM_EPOCHS)
    torch.save(gen.state_dict(), '/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netG_%d.pth' % NUM_EPOCHS)



### TRAINING OF ENCODER
if train_enc:
    # enc_epochs = 5
    kappa = 1.0
    step = 0

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    gen.load_state_dict(torch.load('/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netG_%d.pth' %NUM_EPOCHS))
    gen.eval()
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    critic.load_state_dict(torch.load('/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netD_%d.pth' %NUM_EPOCHS))
    critic.eval()
    for p in gen.parameters():
        p.requires_grad = False
    for p in critic.parameters():
        p.requires_grad = False

    enc = Encoder(Z_DIM, CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(enc)
    criterion = nn.MSELoss()

    # initializate optimizer
    opt_enc = optim.Adam(enc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    padding_epoch = len(str(enc_epochs))
    padding_i = len(str(len(loader)))
    writer_enc = SummaryWriter(f"logs_kernel_3/enc")

    for epoch in range(enc_epochs):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]
            opt_enc.zero_grad()
            z = enc(real)
            fake = gen(z)

            real_features = critic.forward_features(real)
            fake_features = critic.forward_features(fake)

            #izif architecture
            loss_imgs = criterion(fake, real)
            loss_features = criterion(fake_features, real_features)
            enc_loss = loss_imgs + kappa*loss_features

            enc_loss.backward()
            opt_enc.step()
            # if i % CRITIC_ITERATIONS == 0:
            # e_losses.append(e_loss)
        # enc.eval()
        writer_enc.add_scalar('enc_loss', enc_loss.item(), epoch)
        print(f"[Epoch {epoch:{padding_epoch}}/{enc_epochs}] "
                f"[Batch {batch_idx:{padding_i}}/{len(loader)}] "
                f"[E loss: {enc_loss.item():3f}]")
        # step += 1
    torch.save(enc.state_dict(), '/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netE_%d.pth' % enc_epochs)

evaluate = True
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
)
if evaluate:
    criterion = nn.MSELoss()
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    gen.load_state_dict(torch.load('/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netG_%d.pth' %NUM_EPOCHS))
    gen.eval()
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    critic.load_state_dict(torch.load('/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netD_%d.pth' %NUM_EPOCHS))
    critic.eval()
    enc = Encoder(Z_DIM, CHANNELS_IMG, FEATURES_CRITIC).to(device)
    enc.load_state_dict(torch.load('/home/mihael/ML/GANs_n_Anomalies/torch_GAN/f-AnoGAN_w_WGAN-GP/models/netE_%d.pth' %enc_epochs))
    enc.eval()


    with open("score.csv", "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance\n")

    for i, (img, label) in enumerate(test_loader):
        real_img = img.to(device)

        real_z = enc(real_img)  # latent vector
        fake_img = gen(real_z)
        fake_z = enc(fake_img)

        real_feature = critic.forward_features(real_img)  # 1, 256
        fake_feature = critic.forward_features(fake_img)

        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)

        anomaly_score = img_distance + kappa * loss_feature

        z_distance = criterion(fake_z, real_z)

        with open("score.csv", "a") as f:
            f.write(f"{label.item()},{img_distance},"
                    f"{anomaly_score},{z_distance}\n")

        if i % 1 == 0:
            print(f"{label.item()}, {img_distance}, "
                  f"{anomaly_score}, {z_distance}\n")
            compare_images(real_img, fake_img, i, anomaly_score, reverse=False, threshold=0.3)