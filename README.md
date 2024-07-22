
## Galactic Classification

The classification of different types of galaxies is a crucial task in astrophysics that traditionally requires extensive manual effort. With the growing volume of astronomical data captured by modern telescopes, there is a burning need for automated solutions to efficiently and accurately classify galaxies. This research aims to understand and exploit the fundamentals of deep learning, specifically convolutional neural networks (CNNs), to classify various types of galaxies. The proposed methodology includes the design of customised CNN architectures suited to the galaxy classification problem, using transfer learning and fine-tuning with three models: MobileNetV2, EfficientNet and ConvNeXt. The dataset used in this research, known as NA10, consists of a diverse collection of 14,034 galaxy images from Sloan Digital Sky Survey (SDSS). For each of the three classes, 300 images were reserved for the validation and test sets, and the rest were used for training. Significant results are obtained for each architecture, some of them close to the performances achieved in the literature.


## Authors

- [@noah-es](https://github.com/noah-es)


## Deployment

To deploy this project run the FDTD-last.py code.

```bash
  python FDTD-last.py
```


## Take a look

* General Conditions: The proposed example plays with dimensionlessness, to make the code more efficient. In our case, space and time are quantized in quanta of 1 unit, both spatial and temporal. In addition, the speed of light travels through a quantum of space in a quantum of time, so it is worth 1. This greatly simplifies the relationship between c, the wavelength and the frequency of light. 

![General Conditions](Gif/General_Conditions.gif)

* Frequency Conditions: Therefore, for this example, the following scenarios are considered, both divided by the cut-off frequency F<sub>0</sub>.

![Frequency Conditions](Gif/Frequency_Conditions.gif)

* High Frequency: An electromagnetic wave (components of the electric and magnetic field) of frequency 0.21 units can be seen travelling through space, colliding with a barrier (bounded by two vertical bars). The phenomena of reflection and transmission are noteworthy.

![High Frequency](Gif/High_Frequency.gif)

* Low Frequency: An electromagnetic wave (components of the electric and magnetic field) of frequency 0.04 units can be seen travelling through space, colliding with a barrier (bounded by two vertical bars). The phenomena of reflection and transmission are noteworthy.

![Low Frequency](Gif/Low_Frequency.gif)

It is logically observed that the wave with a higher frequency has an electrical size similar to the size of the barrier, which acts as a wall against the advancing electromagnetic wave. The transmission coefficient is very low. The wave practically dissipates after passing through the wall.

 On the other hand, at lower frequencies, the electrical size of the wave is larger than the size of the barrier, thus not being an obstacle and practically not altering the displacement of the wave.

This is a clear example of how WiFis, or radio waves, work.
