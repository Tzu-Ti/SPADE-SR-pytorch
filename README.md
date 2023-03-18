# SPADE-SR-pytorch

The PyTorch implementation of [SPADE-SR](https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Chiu_Low-Resolution_Thermal_Sensor-Guided_Image_Synthesis_WACVW_2023_paper.html).

# Getted Started
```shell
$ git clone https://github.com/Tzu-Ti/SPADE-SR-pytorch.git
```

# Train Generator & Discriminator
```shell
$ python3 train.py --train --model_name D-update2 --d_updates 2 --env D-update2
```
- ```--train``` means train mode, change ```--test``` to test the model
- ```--model_name``` means this model name
- ```--d_updates``` means the number of discriminator optimizer update step
- ```--env``` setting the environment name of visdom

# Train Inversion
```shell
$ python3 train_inversion.py --train --model_name D-update2 --env D-update2
```