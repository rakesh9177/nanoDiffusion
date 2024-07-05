import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class NoiseDiffuser:
    def __init__(self, start_beta, end_beta, total_steps, device):

        assert start_beta < end_beta < 1.0

        self.device = device
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.total_steps = total_steps

        self.betas = torch.linspace(
            self.start_beta, self.end_beta, steps=self.total_steps, device=device
        )

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def noise_diffusion(self, image, t):
        image = image.to(self.device)
        random_noise = torch.randn(image.shape, device=self.device)

        sqrt_alphabar = torch.sqrt(self.alpha_bar)

        processed_image = (
            sqrt_alphabar[t].view(image.shape[0], 1, 1, 1) * image
            + (1 - self.alpha_bar[t].view(image.shape[0], 1, 1, 1)) * random_noise
        )

        return processed_image, random_noise


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """"
      in_channels: input channels of the incoming image
      out_channels: output channels of the incoming image
      """
        super(UNet, self).__init__()

        # Encoder
        self.ini = self.doubleConvolution(inC=in_channels, oC=16)
        self.down1 = self.Down(inputC=16, outputC=32)
        self.down2 = self.Down(inputC=32, outputC=64)

        # ------------------------ Decoder ------------------------#
        self.time_emb2 = self.timeEmbeddings(1, 64)
        self.up2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=2
        )  # 144,912
        self.afterup2 = self.doubleConvolution(inC=64, oC=32)

        self.time_emb1 = self.timeEmbeddings(1, 32)
        self.up1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2
        )
        self.afterup1 = self.doubleConvolution(inC=32, oC=16, kS1=5, kS2=4)

        # ------------------------ OUTPUT ------------------------#
        self.out = nn.Conv2d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x, t=None):
        assert t is not None
        x = x.to(device)
        t = t.to(device)
        # ------------------------ Encoder ------------------------#

        x1 = self.ini(x)  # Initial Double Convolution
        x2 = self.down1(x1)  # Downsampling followed by Double Convolution
        x3 = self.down2(x2)  # Downsampling followed by Double Convolution

        # ------------------------ Decoder ------------------------#
        t2 = self.time_emb2(t)[:, :, None, None]
        y2 = self.up2(x3 + t2)  # Upsampling
        y2 = self.afterup2(
            torch.cat([y2, self.xLikeY(x2, y2)], axis=1)
        )  # Crop corresponding Downsampled Feature Map, Double Convolution

        t1 = self.time_emb1(t)[:, :, None, None]
        y1 = self.up1(y2 + t1)  # Upsampling
        y1 = self.afterup1(
            torch.cat([y1, self.xLikeY(x1, y1)], axis=1)
        )  # Crop corresponding Downsampled Feature Map, Double Convolution
        outY = self.out(y1)  # Output Layer (ks-1, st-1, pa-0)

        return outY

    # --------------------------------------------------------------------------------------------------- Helper Functions Within Model Class

    def timeEmbeddings(self, inC, oSize):
        """
      inC: Input Size, (for example 1 for timestep)
      oSize: Output Size, (Number of channels you would like to match while upsampling)
      """
        return nn.Sequential(nn.Linear(inC, oSize), nn.ReLU(), nn.Linear(oSize, oSize))

    def doubleConvolution(self, inC, oC, kS1=3, kS2=3, sT=1, pA=1):
        """
      Building Double Convolution as in original paper of Unet
      inC : inputChannels
      oC : outputChannels
      kS1 : Kernel_size of first convolution
      kS2 : Kernel_size of second convolution
      sT: stride
      pA: padding
      """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=inC,
                out_channels=oC,
                kernel_size=kS1,
                stride=sT,
                padding=pA,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=oC,
                out_channels=oC,
                kernel_size=kS2,
                stride=sT,
                padding=pA,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def Down(self, inputC, outputC, dsKernelSize=None):
        """
      Building Down Sampling Part of the Unet Architecture (Using MaxPool) followed by double convolution
      inputC : inputChannels
      outputC : outputChannels
      """

        return nn.Sequential(
            nn.MaxPool2d(2), self.doubleConvolution(inC=inputC, oC=outputC)
        )

    def xLikeY(self, source, target):
        """
      Helper function to resize the downsampled x's to concatenate with upsampled y's as in Unet Paper
      source: tensor whose shape will be considered ---------UPSAMPLED TENSOR (y)
      target: tensor whose shape will be modified to align with target ---------DOWNSAMPLED TENSOR (x)
      """
        x1 = source
        x2 = target
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x1


import torchvision
import torchvision.transforms as transforms

batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and Load the MNIST dataset
transform = transforms.ToTensor()
full_trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Splitting the trainset into training and validation datasets
train_size = int(0.8 * len(full_trainset))  # 80% for training
val_size = len(full_trainset) - train_size  # remaining 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(
    full_trainset, [train_size, val_size]
)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
valloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    diffuser,
    totalTrainingTimesteps,
):
    """
    model: Object of Unet Model to train
    train_loader: Training batches of the total data
    val_loader: Validation batches of the total data
    optimizer: The backpropagation technique
    criterion: Loas Function
    device: CPU or GPU
    num_epochs: total number of training loops
    diffuser: NoiseDiffusion class object to perform Forward diffusion
    totalTrainingTimesteps: Total number of forward diffusion timesteps the model is to be trained on
    """

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_train_loss = 0

        # Wrapping your loader with tqdm to display progress bar
        train_progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False,
        )
        for batch_idx, (data, _) in train_progress_bar:
            data = data.to(device)

            # Use a random time step for training
            batch_size = len(data)
            timesteps = (
                torch.randint(0, totalTrainingTimesteps, (batch_size,), device=device)
                .long()
                .tolist()
            )

            noisy_data, added_noise = diffuser.noise_diffusion(
                data, torch.tensor(timesteps)
            )

            predicted_noise = model.forward(
                x=noisy_data,
                t=torch.tensor(timesteps).to(torch.float32).to(device).view(-1, 1),
            )
            loss = criterion(added_noise, predicted_noise)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds")
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        # Wrapping your validation loader with tqdm to display progress bar
        val_progress_bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
            leave=False,
        )
        with torch.no_grad():
            for batch_idx, (data, _) in val_progress_bar:
                data = data.to(device)

                # For simplicity, we can use the same random timestep for validation
                batch_size = len(data)
                timesteps = (
                    torch.randint(
                        0, totalTrainingTimesteps, (batch_size,), device=device
                    )
                    .long()
                    .tolist()
                )

                noisy_data, added_noise = diffuser.noise_diffusion(data, timesteps)
                predicted_noise = model(
                    x=noisy_data,
                    t=torch.tensor(timesteps).to(torch.float32).to(device).view(-1, 1),
                )

                loss = criterion(added_noise, predicted_noise)
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses


total_timesteps = 1000
startBeta, endBeta = 1e-4, 0.02
inputChannels, outputChannels = 1, 1
num_epochs = 50
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


stableDiffusionModel = UNet(in_channels=inputChannels, out_channels=outputChannels)
stableDiffusionModel = stableDiffusionModel.to(device)
optimizer = torch.optim.AdamW(
    stableDiffusionModel.parameters(), lr=1e-3, weight_decay=0.9999, fused=True
)
criterion = nn.MSELoss()
diffuser = NoiseDiffuser(startBeta, endBeta, total_timesteps, device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_params = count_parameters(stableDiffusionModel)
print(f"The model has {num_params:,} trainable parameters.")
############################################################################
#                                 TO DO                                    #
#                Execute this Block, Train & Save the Model                #
#                            And Plot the Progress                         #
############################################################################

train_losses, val_losses = train(
    model=stableDiffusionModel,
    train_loader=trainloader,
    val_loader=valloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=num_epochs,
    diffuser=diffuser,
    totalTrainingTimesteps=total_timesteps,
)

# Save the model
torch.save(stableDiffusionModel.state_dict(), "HW3SDModel.pth")

# Plot the losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
