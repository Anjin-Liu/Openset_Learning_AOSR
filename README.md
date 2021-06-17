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