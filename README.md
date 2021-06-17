# Open_Set_Learning_AOSR
Traditional supervised learning aims to train a classifier in the closed-set world, where training and test samples share the same label space. In this paper, we target a more challenging and realistic setting: open-set learning (OSL), where there exist test samples from the classes that are unseen during training. Although researchers have designed many methods from the algorithmic perspectives, there are few methods that provide generalization guarantees on their ability to achieve consistent performance on different training samples drawn from the same distribution. Motivated by the transfer learning and probably approximate correct (PAC) theory, we make a bold attempt to study OSL by proving its generalization error−given training samples with size n, the estimation error will get close to order Op(1/√n). This is the first study to provide a generalization bound for OSL, which we do by theoretically investigating the risk of the target classifier on unknown classes. According to our theory, a novel algorithm, called auxiliary open-set risk (AOSR) is proposed to address the OSL problem. Experiments verify the efficacy of AOSR.


传统的监督学习假设训练数据集和测试数据集共享相同的标签空间。但我们并不知道测试数据集的真实标签。在测试过程中测试数据集也包含一些额外的类别。这种情况也叫做开放集假设。为了解决监督问题做开放集假设下的影响，开放集学习被提出并被很多学者研究了。以前的几乎所有方法都只考虑开放集学习的算法，并未对开放集学习问题本身的可解性进行更深入的思考。本质上开放集学习可解性是一个数学问题。本文构建了第一个开放集学习的数学理论去解决该问题。我们的数学理论主要使用了PAC学习理论和迁移学习理论。理论结果显示:在一定假设下，存在一个开放集学习算法，使得该算法的泛化误差达到Op(1/√n). 我们基于理论也设计了开放集学习算法。该算法主要通过构造辅助数据去挖掘额外类别信息。实验也表明在许多基础数据集上，我们的算法实现好的效果。

The contribution of this paper is to build the open-set learning theory. Under suitable assumptions, we provide the generalization error under the empirical risk minimization principle. Our theory shows that open-set learning is solvable with suitable assumptions. We think this is important and novel for open set learning. We list several interesting and important problems for open-set learning theory as follows.
1. How to construct suitable neural networks to approximate assumption 1 ? This problem has been solved by the author. The answer is positive.
2. How to construct weaker assumption to replace assumption realization for achieving similar results ? 
3. Without assumption realization, what will happen  ? 
4. Is it possible for OSL to achieve agnostic PAC learnability and achieve  fast learning rate $O_p(1/{n}^{a})$, for $a>0.5$ ?  
5. Is it possible to construct OSL learning theory by  stability theory  ?

本文的核心贡献是创立了开放集学习的学习理论。该理论的目的是探讨开放集学习的可解性问题。在适当的假设下，我们给出了开放集学习的经验风险极小化的泛化误差。这也表明开放集学习在一定条件下其本身是可解的。未来还需要怎么发展开放集学习理论呢？
1.找的合适神经网络结构去逼近文中假设1。该问题容易解决并且已被解决，但并未放到文中。存在神经网络结构逼近假设1。
2.去掉假设1，会有什么情况发生？这问题难度中等，可被快速解决。
3.开放集学习的PAC可学习性问题？文中只证明了近乎PAC可学习性，但PAC可学习性并未被彻底解决。个人觉得该问题并不具有PAC可学习性。因为已知类和未知类边界是关键的阻碍。
4.快速学习率问题。是否存在开放集学习算法，其样本复杂度比O（1/sqrt(n)）小？这个问题是有趣的并且也是重要的。
5.用稳定性理论去构建开放集学习理论。本文构建开放集学习理论主要利用了经验风险极小化。那么怎么用稳定性理论去构建开放集学习理论呢？

Different with other algorithms
To recognize the unknown samples, our algorithm AOSR depends on the score of hypothesis function, i.e., the label with largest score is regarded as the predicted label. Hence, it is different with many previous algorithms, which use a fix threshold to recognize the unknown samples. However, AOSR has different thresholds for different samples. 

与其他方法的差别
我们的算法是基于分类器的，也就是输出个各类的概率值（其中包括未知类），通过概率大小选择预测的类别。所以本质上，我们的方法对不同的样本都会自动生成一个阈值。而很多其他方法使用固定阈值去辨识未知样本。 

Evaluation
The macro-average F1 scores are used to evaluate OSL. The area under the receiver operating characteristic (AUROC) is also frequently used. Note that AUROC used in many open-set learning methods is suitable for global threshold-based OSL algorithms that recognize unknown samples by a fix threshold. However, AOSR recognizes unknown samples based on the score of hypothesis function, thus, AOSR has different thresholds for different samples. This implies that AUROC is not suitable for our algorithm. In this paper, we use macro-average F1 scores to evaluate our algorithm.

We believe that how to design a suitable OSL evaluation is still an open problem !  

评估方式
F1 scores 是一个常用的方式去评估开放集学习的效果。AUROC也是一个常用的方式。许多基于固定阈值的方法会使用AUROC。因为通过AUROC，这些方法能够评估出阈值选择的鲁棒性。但我们的方法并不适合AUROC。因为不同于以往通过固定阈值选择去未知类的算法，我们的算法是基于分类器的，也就是输出个各类的概率值（其中包括未知类），通过概率大小选择预测的类别。所以本质上，我们的方法对不同的样本都会自动生成一个阈值。那么在这种情况下，用以前开放集学习常用的AUROC作为衡量方法对我们方法是不合理的。所以在本文中我们用F1 scores去评估开放集学习的效果。

我们认为设计一个合适的开放集学习的评估方法仍是一个公开问题！
