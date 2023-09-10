---
title: 【Nori】Monte Carlo Sampling and Ambient Occlusion
date: 2023-09-04 11:18:34
toc: true
mathjax: true
categories: Computer Graphics
tags:
    - Sampling
    - Nori
---
这次的作业分为两部分：

1. 实现一些蒙特卡洛的采样算法
2. 实现点光源和环境光遮蔽


借着这个机会让我们先复习一下蒙特卡洛积分：


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


# 抽样随机变量

在蒙特卡洛积分的框架中，合适的采样策略能够显著加快收敛速度。一个核心的研究问题是如何实现满足分布q(x)的随机变量的采样。尽管存在多种策略，但在此，我们将专注于PBRT所描述的“反演法”。

为了深入理解此方法，首先考虑离散概率分布。假设存在一个分布，其概率之和为1，其概率密度函数(pdf)与累积分布函数(cdf)如下

|pdf|cdf|
|---|---|
|![](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/discrete-pdf.svg)|![](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/discrete-cdf.svg)|

从p(x)分布中抽取一个随机变量的过程可以可以通过如下方法获得：

1. 计算CDF: $P(x) = \int_{0}^{x} p(x^{'})dx^{'}$
2. 计算CDF的逆函数 $P^{-1}(x)$
3. 从均匀分布的随机数生成器抽取一个随机变量$\xi$
4. 计算随机变量$X_i = P^{-1}(\xi)$

具体过程如下图所示：

![](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/discrete-inversion.svg)

值得注意的是，虽然此例为离散情况，但该策略在连续分布情境中同样有效。


# 蒙特卡洛采样
了解了基础知识后，让我们正式进入Nori的作业部分

## Tent
$$
p(x,y) = p_1(x)p_1(y),\space p_1(t) = \begin{cases}
1 - \mid t \mid, &\quad -1 \le t \le 1 \newline
0 &\quad \text{otherwise}
\end{cases}
$$
我们先来求出cdf, 分段积分就好：

$$
P_i(t)  = \begin{cases}
0, &\quad  t < -1 \newline
\frac{1}{2}(t+1)^2, &\quad -1 \le t < 0 \newline
\frac{1}{2}(t-1)^2, &\quad 0 \le t \le 1 \newline
1 &\quad t > 1
\end{cases}
$$
将$\xi$代入得：

$$
P_i^{-1}(\xi)  = \begin{cases}
\sqrt{2} - 1, &\quad 0 \le \xi < \frac{1}{2} \newline
1 - \sqrt{2 -2t}, &\quad \frac{1}{2} \le t \le 1 \newline
\end{cases}
$$


## Uniform Disk
这个问题需要我们在圆盘上均匀采样，所以我们有：
$$
p(x,y) = \frac{1}{\pi}
$$
因为要在圆盘上积分，所以我们可以换元到极坐标上：
$$
p(r, \theta) = p(x,y)r = \frac{r}{\pi}
$$
根据上文的公式我们有:
$$
\begin{align}
    &p(r) = \int_0^{2\pi} p(r, \theta) d\theta = 2r \newline
    &p(\theta \mid r) = \frac{p(r, \theta)}{p(r)} = \frac{1}{2\pi}
\end{align}
$$
将$\xi$代入得：
$$
\begin{align}
    &r = \sqrt{\xi_1} \newline
    &\theta = 2\pi\xi_2
\end{align}
$$


## Uniform Sphere/ Uniform Sphere
半球和球的做法几乎一样，我们就以半球积分为例子。

根据半球的表面积公式，我们可以得到：
$$
p(\omega) = \frac{1}{2\pi}
$$
换元到球座标上
$$
p(\theta, \phi) = \frac{sin(\theta)}{2\pi}
$$
我们可以得到$\theta$和$\phi$的pdf:
$$
\begin{align}
 &p(\theta) = \int_0^{2\pi} p(\theta, \phi) d\phi = sin(\theta) \newline
 &p(\phi \mid \theta) = \frac{\theta, \phi}{p(\theta)} = \frac{1}{2\pi}
\end{align}
$$
下面来求cdf:
$$
\begin{align}
 &P(\theta) = \int_0^{\theta} sin(\theta^{'}) d\theta^{'} = 1 - cos\theta \newline
 &P(\phi \mid \theta) = \int_0^{\phi} \frac{1}{2\pi} d \phi^{'} = \frac{\phi}{2\pi}
\end{align}
$$
将$\xi$代入得：
$$
\begin{align}
 &\theta = acos \xi_1 \newline
 &\phi = 2\pi \xi_2
\end{align}
$$
大部分时候我们需要返回笛卡尔坐标：

$$
\begin{align}
 &x = sin(\theta)cos(\phi) = cos(2\pi \xi_2) \sqrt{1 - \xi_1^2}\newline
 &y = sin(\theta)sin(\phi) = sin(2\pi \xi_2) \sqrt{1 - \xi_1^2} \newline
 &z = cos\theta = \xi_1
\end{align}
$$

## Cosine Hemisphere
在这道题中我们希望概率分布为$p(\omega) = p(\theta) = \frac{cos\theta}{\pi}$, 那么我们就可以通过换元得到
$$
p(\theta, \phi) = \frac{cos\theta sin \theta}{\pi}
$$
虽然之前的反演法也可以求这个分布，但我们有更简单的方法，“Malley’s method”。
具体地说，该方法首先在半圆的投影，即单位圆盘上，进行均匀采样以获得x和y坐标。接着，使用圆盘上的点到半球的距离$sin(\theta)$ 作为z 轴坐标。

如下图所示：
![](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Malleys%20method.svg)

下面我们需要证明的就是有上述方法采样得到的sample：$(x,y,sin(\theta))$的分布符合要求

我们有圆盘上的均匀概率密度为$p(r,\theta) = \frac{r}{\pi}$.

考虑如下变换:$ (\theta,\phi) \rightarrow (r, \phi) $
我们有：
$$
\begin{cases}
 &r = sin(\theta) \newline
 &\phi = \phi
\end{cases}
$$
所以概率密度的变化为：
$$
\begin{align}
p(\theta,\phi) &= \begin{vmatrix} cos\theta & 0 \newline 0 & 1 \end{vmatrix} * p(r, \phi) \newline
               &= cos\theta * p(r, \phi) \newline
               & = \frac{cos\theta sin \theta}{\pi}
\end{align}
$$
于是我们可以直接引用之前实现的圆盘采样：

```c++
Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
    Point2f disk = squareToUniformDisk(sample);
    float x = disk.x();
    float y = disk.y();
    return Vector3f(x, y, sqrt(1 - x * x - y * y));
}

```
