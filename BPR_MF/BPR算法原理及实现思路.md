# BPR算法原理及实现思路

[TOC]

## 1.背景介绍：

BPR(Bayesian Persionalized Ranking)贝叶斯个性化排序，是一种在实际场景中运用非常广泛的排序算法。

排序算法：1.pointwise approach:点对排序算法，即常规的点击率经常用到的0，1预测问题；2.pairwise approach	: 对排序算法，排序被转化为对序列分类或者对序列回归问题。3.listwise approach:列表排序算法，在学习和预测过程中都将排序列表作为一个样本，排序的结构被保留。

常规排序优化思路是对于一个给定的数据集例如movielens这里数据是用户对电影的评分，用户对一个电影打过分我们认为用户喜欢该电影，反之我们认为用户不喜欢，这样我们就能构造user，item行为对，然后用machine learning算法进行优化，来预测用户对那些没有行为的item的概率，最后更加这个概率进行排序推荐。这种方式理论上是可行的，但是有两个问题无法解决：1.对于用户来说，大多数的item是没有行为的，如果直接认为这些都是不喜欢是不合理的，因为有些电影用户喜欢可能用户之前已经看过就没有在看，有些确实是用户不喜欢的，这样训练模型会导致一个非常严重的问题就是我们训练数据中绝大多数都是负样本，模型学习到的就是0，想象一下如果模型全部预测为0结果也不会很差，因为大多数数据的target是0；2.我们这种优化方式是预测用户喜不喜欢，并没有去优化用户到底有多喜欢或者说用户喜欢各个商品的程度，因此这个概率不具有排序的意义。



## 2.BPR算法

针对上述问题，作者这里提出了基于贝叶斯排序的排序算法框架，这里作者的假设是用户对于那些评过分的电影的喜好程度大于没有评过分的，意思就是作者这里构建了一个item pair对，<u,i,j>表示用户对i的喜好程度强于j这样模型优化的结果就是用户对于两个item的喜好程度。

贝叶斯排序算法顾名思义，是基于贝叶斯后验概率的算法
$$
P(\theta|>_u) = \frac{P(>_u|\theta)P(\theta)}{P(>_u)}=P(>_u|\theta)P(\theta)
$$
$>_u​$是用户期望的潜在偏好结构，作者作了两个假设，首先假设用户的行为彼此是独立的，同时假设用户的每对item对之间是独立的因此上述表达式中$P(>_u|\theta)​$可以改写为
$$
\prod_{u\in U}{P(>_u|\theta)}=\prod_{(u,i,j)\in U\times I \times I}{P(i>_u j|\theta)^{\delta{(u,i,j)\in Ds}}}{(1-P(i>_u j|\theta))^{\delta{(u,i,j)\notin Ds}}}
$$
其中$\delta$是指示函数当$\delta(u,i,j) \in Ds$时为1反之为0，由于i,j对的对称性，因此上述等式的后半部分可以省略即表达式为
$$
\prod_{u\in U}{P(>_u|\theta)}=\prod_{(u,i,j)\in U\times I \times I}{P(i>_u j|\theta)}
$$
由于$P(i>_u j|\theta)$是一个概率这里可以等价于$\delta(X_{uij}(\theta))$这里外层是一个sigmoid函数将概率归一到0-1。

上述过程总结一下可以得处BPR-opt的优化过程
$$
BPR-opt:=ln(P(\theta|>_u))=ln(P(>_u|\theta)*P(\theta))=ln\prod\delta(x_{uij}(\theta))P(\theta)=\sum_{(u,i,j)\in D_s}{ln(\delta(x_{uij}))+ln(P(\theta))}
=\sum_{(u,i,j)\in D_s}{ln(\delta(x_{uij}))-\lambda_{\theta}||\theta||^2}
$$
根据上述表达式这里显然我们对于不同的任务只需要寻找$x_{uij}$的表达式。对于一般任务最简单的做法是用$x_{ui}-x_{uj}$



代码实现: