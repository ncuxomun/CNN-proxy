# CNN-proxy - Convolutional Neural Network as a forward model approximation 
# Optimization Project
In this repository I present a simple yet very compact way of setting a [proxy model](https://github.com/ncuxomun/CNN-proxy/blob/main/proxy_model_64.py) based on PyTorch. PyTorch Lighnintng helps to keep the setup compact and neat.

Brief description of the model:
  - The proxy is constructed in such way that takes (unpacked) [channelized (binary) maps](https://github.com/ncuxomun/CNN-proxy/blob/main/channels_uncond_10k.7z) and predicts normalized oil rates, water cut values, and flowing bottom-hole pressures. 
  - The key idea is to map the complex geologic features to respective production data profiles.
  - The geological model is a 2D map that is 64x64 pixel sized, which accomodates two producing and injective wells.

The following figures present prediction results for a model that was trained on 2000 unconditional samples, where 80% of data was set for training and 20% of data was set for validation. The figures below present some results given nine input maps with their respective predictions (in blue) compared with true (in red) values. Note that both input and output data were normalized from 0 to 1 for training purposes.
Basically, the production profiles correspond to respective tile in figure Maps. Please note that for compactness, I combined production (oil rate and watercut) profiles for two production wells in a single tile by concatenating vectors of size 25x1. Similarly, I combined injection profiles for two injector wells in a single tile by concatenating vectors of size 25x1.

![myimage-alt-tag](https://github.com/ncuxomun/CNN-proxy/blob/main/1.png)
![myimage-alt-tag](https://github.com/ncuxomun/CNN-proxy/blob/main/2.png)
![myimage-alt-tag](https://github.com/ncuxomun/CNN-proxy/blob/main/3.png)
![myimage-alt-tag](https://github.com/ncuxomun/CNN-proxy/blob/main/4.png)
