# ICML2021
# Abstract
Traditional supervised learning aims to train a classifier in the closed-set world, where training and testing samples share the same label space. In this paper, we target a more challenging and realistic setting: open-set learning (OSL), where there exist testing samples from the classes that are unseen during training. 
Although researchers have designed many methods from the algorithmic perspectives, there are few methods that provide generalization guarantees on their ability to achieve consistent performance on different training samples drawn from the same distribution. Motivated by the transfer learning and probably approximate correct (PAC) theory, we make a bold attempt to study OSL by proving its generalization error$-$given  training samples with size $n$, the estimation error  will get close to order $$O_p(1/\sqrt{n})$$.
This is the first study to provide a generalization bound for OSL, which we do by theoretically investigating the risk of the target classifier on unknown classes. According to our theory, a novel algorithm, called  auxiliary open-set risk (AOSR) is proposed to address the OSL problem. Experiments verify the efficacy of AOSR and support our theory.

## Intuitive Illustration
<img src="https://raw.githubusercontent.com/Anjin-Liu/Openset_Learning_AOSR/master/assest/moon_train_data.jpg" width="312" />
(a) Moon shape toy data for training

<img src="https://raw.githubusercontent.com/Anjin-Liu/Openset_Learning_AOSR/master/assest/moon_closeset_pred.jpg" width="312" />
(b) Moon shape toy data close-set decision region

<img src="https://raw.githubusercontent.com/Anjin-Liu/Openset_Learning_AOSR/master/assest/moon_openset_pred.jpg" width="312" />
(c) Moon shape toy data open-set decision region

Our solution not just learn the **decision boundary** as what closed-set learning do, but also learn the support set so that the **decision region** can be built as a closure, as shown in figure (a), (b) and (c).

## For Reproducing the Experiment Results
All the running details can be checked in this repo.
Simply open the **.ipynb** file, the experiment details will be displayed.


To reproduce the experiment, few python package dependencies are required.
For simplicity, we display all the running details in jupyter notebooks.
The output of each stage is clearly shown.
### requirements:
    pytoch
    tensorflow
    pandas
    numpy
    scikit-learn

**aosr_utility.py** contains the **AOSR loss function** for tensorflow and the **isolation forest based open-set sample enrichment function**

**mnist_exp_showcase.ipynb** presents the running details of AOSR on MNIST, Omniglot, MNIST-noise, and noise dataset

**cifar10_exp_showcase.ipynb** presents the running details of AOSR on cifar10, ImageNet Resize, ImageNet Crop, LSUN Resize and LSUN Crop dataset

## Neural Network Structure

### Double Moon:
**Double Moon Encoder**: None, scince the input space is $$\mathbb{R}^2$$
**Double Moon Open-set Learning Neural Network**:

```
Dense(64, activation='relu'),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(32, activation='relu'),
Dense(16, activation='relu'),
Dense(16, activation='relu'),
Dense(8, activation='relu'),
Dense(8, activation='relu'),
Dense(3),
Activation(activation='softmax')
```
To Initialize:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
To Learn
```
optimizer='adam',
loss=pq_risk(detetor, z_q_sample, z_q_weight, z_p_X, 0.15, 2),
learning_rate=0.001 for 20 epochs
learning_rate=0.0001 for 10 epochs
```

### MNIST
**MNIST Encoder**: Plain CNN 
```
Conv2D(filters=100, kernel_size=(3, 3),activation="relu"),
.Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
MaxPooling2D(pool_size=(2, 2)),
Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(500),
Dense(10),
Activation(activation='softmax')
```
To Encode:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
**MNIST Open-set Learning Neural Network**:
```
Dense(11),
Activation(activation='softmax')
```
To Initialize:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
To Learn
```
optimizer='adam',
loss=pq_risk(detetor, z_q_sample, z_q_weight, z_p_X, 0.15, 2),
learning_rate=0.001
epochs=25
```
### CIFAR10
**CIFAR10 Encoder**: ResNet18
To Encode:
```
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```
**CIFAR10 Open-set Learning Neural Network**:
```
Dense(11),
Activation(activation='softmax')
```
To Initialize:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
To Learn
```
optimizer='adam',
loss=pq_risk(detetor, z_q_sample, z_q_weight, z_p_X, 0.15, 2),
learning_rate=0.001
epochs=30
```