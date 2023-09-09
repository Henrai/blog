---
title: Nori3
date: 2023-09-04 11:18:34
tags:
mathjax: true
---

# 蒙特卡洛积分
 在解决渲染问题的时候我们需要计算很多积分，比如大家都耳熟能详的渲染方程：
 $$
 L_0(x,\vec{\omega}) = L_e(x,\vec{\omega}) + \int_\Omega f_r(x, d\vec{\omega}^{'}, d\vec{\omega}) L_i(x, d\vec{\omega}^{'}) d\vec{\omega}^{'}
 $$

在数值分析中，我们学习了多种积分算法，如梯形法、插值法和高斯求积法等。这些方法对于一元积分都能简单且高效地求解。但随着维度的增加，积分区间变得越来越复杂，这些传统方法就不再那么适用了。尤其在图形学中，我们经常需要处理像渲染方程这样的高维积分问题。因此，我们需要一种其实现复杂度和收敛速度与维度无关的方法——**蒙特卡洛积分**。

蒙特卡洛(Monte Carlo) 算法是一种随机算法。它通过多次随机采样来逼近所求的结果。当重复的抽样次数足够多时，得到的平均值在统计意义上可以逼近真实的值。而蒙特卡洛积分就是使用蒙特卡洛算法来进行积分的计算。


# Monte Carlo Estimator
根据蒙特卡罗方法的定义，我们可以得到如下的Estimator:
$$
F_n = \frac{1}{n} \sum_{i=1}^{n} \frac{f(X_i)}{p(X_i)}
$$

$F_n$ 的期望就是所求的积分的值，证明过程如下：

$$
\begin{align*}
E[F_n] & =  E[\frac{1}{n} \sum_{i=1}^{n} \frac{f(X_i)}{p(X_i)}] \newline
       & = \frac{1}{n} \sum_{i=1}^{n} \int_a^b \frac{f(x)}{p(x)} p(x) dx \newline
       & = \frac{1}{n} \sum_{i=1}^{n} \int_a^b f(x) dx \newline
       & = \int_a^b f(x) dx
\end{align*}
$$

我们可以通过计算方差来估计蒙特卡洛积分的误差：
$$
\begin{align*}
V[F_n] & = V[ \frac{1}{n} \sum_{i=1}^{n} \frac{f(X_i)}{p(X_i)}] \newline
       & = \frac{1}{n^2} V[\sum_{i=1}^{n} \frac{f(X_i)}{p(X_i)}] \newline
       & =  \frac{1}{n^2} \sum_{i=1}^{n} V[ \frac{f(X_i)}{p(X_i)}] \newline
       & = \frac{1}{n} V[ \frac{f(X_i)}{p(X_i)}]
\end{align*}
$$
所以蒙特卡洛积分器的标准误差为
$$
    SE = \frac{V}{\sqrt{n}} \sim O(n^{-1/2})
$$

# 重要性采样

蒙特卡洛（Monte Carlo）方法提供了一种无偏估计的机制，这意味着给定充分的采样数量，该方法可以确保逼近真实的期望值。但是，其收敛性受到$O(n^{-1/2})$ 的限制，因此在大量采样情况下，其精度的增加速度较慢。为了提高这种估计的效率，研究者们探索了方差减少（Variance Reduction）策略。其中，重要性采样（Importance Sampling）是一种广泛应用的方差减少方法。它旨在通过更精确地对重要区域进行采样（即函数值较大的区域）来优化采样分布，从而提高估计的精度。

我们可以通过以下方式进行简要证明：

假定我们希望估算函数$f(x)$关于概率密度函数$p(x)$的期望值：
$$
I = E[f(X)] = \int f(x)p(x) dx
$$

考虑到$f(x)$呈正态分布特性，当我们采用均匀的概率密度函数$p(x)$进行采样时，方差可能会变得较大。例如，在函数的一倍标准差范围内的采样概率与两倍标准差之外的采样概率相同。

为了优化这一估计，我们可以引入另一个概率密度函数$q(x)$来进行采样，满足$\int q(x) dx = 1$。于是，我们的期望值公式可以重写为：

$$
\begin{align*}
I  & = E[f(X)] \newline
   & = \int f(x)p(x) dx \newline
   & = \int f(x)\frac{p(x)}{q(x)}q(x) dx \newline
   & = E_q[f(X) \frac{p(x)}{q(x)}] \newline
\end{align*}
$$


其对应的方差为：
$$
V_q[f(X) \frac{p(x)}{q(x)}] = E_q[(f(x)\frac{p(x)}{q(x)})^2] - I^2
$$

为了使得新的方差低于原来基于$p(x)$的方差，理想的$q(x)$应与$f(x)p(x)$成正比。当$q(x)$在函数值较大的区域有更高的权重时，我们可以更有效地降低方差，从而获得更优的估计结果。


# 分布变换

我们已经知道通过合理的采样可以加速蒙特卡洛积分的收敛，但是在程序中我们只能
