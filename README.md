# Bird Song ConvGRU
> Using a CNN + ConvGRU to identify bird songs as spectrograms.

This project combines CNNs and GRU's to classify bird species
from spectrograms. 

The model consists of a ResNet18 model with 
the last two layers, AdaptiveAvgPooling and Linear, removed to 
preserve some dimensionality of the extracted features. This 
CNN backbone extracts a volume of features for some N 
number of frames. Those N feature volumes are sent through as a sequence to a 
ConvGRU model to leverage the temporal nature of spectrograms
to produce a classification. 