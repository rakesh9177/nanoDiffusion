# nanoDiffusion
This is inspired from nanogpt to build a smaller and basic version of stable diffusion, hence named nanodiffusion. 
This is a very basic implementation of diffusion models based on ddpm paper, it uses a basic unet with residual networks and noise diffuser based of ddpm paper which focus more on implementation of forward and reverse diffusion with pytorch optimizations for training and sampling

DDPM paper - https://arxiv.org/pdf/2006.11239

Implemention includes:

1. Fixed forward process - adding noise to images
2. UNet - for learning noise from data
3. Reverse Diffusion - for creating images from noise

Optimizations:

1. Normalization - When any normalization is used(group or batch), it cancels out bias terms, therefore no bias term is needed in convolution layers effectively reducing number of parameters
2. Fused AdamW optimizer is used for efficient training

Further improvements that can be done:

1. Unet with attention layers will work better and generate coherent images
2. Text support with CLIP can be added to text-to-image
3. Other sampling methods can be used

