---
title: Reflection and Refraction
date: 2023-10-04 22:08:06
mathjax: true
tags:
- physics
---



## 边界条件：

### 电场
为了方便计算，我们可以把电场分解成垂直和平行两个方向的分量：$E_{\parallel}$ 和 $E_{\perp}$

考虑两种介质之间的界面。设介质1位于界面上方，介质2位于下方。如下图所示， 让我们来计算一下圆柱形所包围的电通量

根据介质中的高斯定理我们有
$$
\oint \mathbf{E} \cdot d\mathbf{A} = \frac{\sum Q}{ｋ_e \varepsilon_0}
$$
因为介质中没有自由电荷，所以我们有
$$
 ｋ_e \varepsilon_0 \oint \mathbf{E} \cdot d\mathbf{A} = 0
$$
带入所有电场分量我们有:
$$
ｋ_{e1} \varepsilon_0 E_{1 \perp } - ｋ_{e2} \varepsilon_0 E_{2 \perp}  = 0
$$
所以我们能得到：
$$
k_{e1} E_{1\perp} = k_{e2} E_{2 \perp} \quad \text{... (1)}
$$


现在，考虑法拉第定律， 如下图所示的环路积分为：
$$
\oint \mathbf{E} \cdot d\mathbf{l} = \frac{\partial \Phi_\mathbf{B}}{\partial t}
$$
假设环路的高 $d \mathbf{l}$ 无限趋近与0， 那么磁通量也为0， 则：
$$\oint \mathbf{E} \cdot d\mathbf{l} = 0$$
所以我们有：
$$
E_{1t} = E_{2t} \quad \text{... (2)}
$$



<table>
<tr>
<th> Gauss's Law </th>
<th> Faraday's law </th>
</tr>
<tr>
<th><img src="maxwell1.png" width="400px"></th>
<th><img src="maxwell2.png" width="400px"> </th>
</tr>
</table>



## 磁场
因为没有磁单极，我们知道磁场 $\mathbf{B}$ 的散度恒为零：
$$
\nabla \cdot \mathbf{B} = 0
$$
这意味着交界面上的磁场 $\mathbf{B}$ 的法线分量是连续的，即：
$$
B_{1 \perp} = B_{2 \perp} \quad \text{... (3)}
$$

根据安培的环路定律，我们有：
$$
\oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} + \mu_0 \varepsilon_0  \frac{\partial \Phi_\mathbf{E}}{\partial t}
$$
因为电介质里没有电流所以我们能得到：
$$
\oint \mathbf{B} \cdot d\mathbf{l} = + \mu_0 \varepsilon_0  \frac{\partial \Phi_\mathbf{E}}{\partial t}
$$
在通过使用求解法拉第定律方程类似的方法，我们可以将电通量也清零，于是我们有：
$$
\oint \mathbf{B} \cdot d\mathbf{l} = 0
$$
带入所有磁场分量我们有：
$$
H_{1 \parallel} = H_{2 \parallel} \quad \text{... (4)}
$$

总结一下我们得到的所有边界条件：

$$
\begin{align*}
& k_{e1} E_{1\perp} = k_{e2} E_{2 \perp} && \quad \text{... (1)} \\
& E_{1t} = E_{2t} && \quad \text{... (2)} \\
& B_{1 \perp} = B_{2 \perp} && \quad \text{... (3)} \\
& H_{1 \parallel} = H_{2 \parallel} && \quad \text{... (4)}
\end{align*}
$$


# Fresnel

有了边界条件后我们就可以正式进入Fresnel 方程的推导。在这里，我们同样把电场分为垂直与纸面的 $E_{\perp}$ 和平行于直面的 $E_{\parallel}$

<img src="fresnel1.png" width="300px">

让我们先考虑 $E_{\perp}$:
因为 $E_{\perp}$ 平行于平面，所以根据(2) 我们有：
$$E_i + E_r = E_t \text{... (i)}$$
我们可以把与电场垂直的磁场再次分为  $B_{\perp}$ 和 $B_{\parallel}$
然后根据(4), 我们有：
$$ B_i cos i - B_r cos r = B_t cos t \quad \text{... (ii)}$$
我们可以通过Maxwell方程得到B和E的关系为:
$$
B = \frac{E}{v} = \frac{nE}{c}
$$
代回原式可得：
$$
  n_1 E_i cos i - n_1 E_r cos r = n_2 E_t cos t \quad \text{... (iii)}
$$
由反射折射的性质我们可以知道:
$$i = r = \theta_i, t = \theta_t$$
再将(i) 代入 (iii) 得：
$$
  n_1 E_i cos \theta_i - n_1 E_r cos \theta_i = n_2 (E_i + E_r) cos \theta_t
$$

整理得：
$$
r^{\perp} = \frac{E_r^{\perp}}{E_i^{\perp}} = \frac{n_1 cos \theta_i - n_2 cos \theta_t} {n_1 cos \theta_i + n_1 cos \theta_t}
$$


通过类似的方法我们还可以得出：

$$
r^{\parallel} = \frac{E_r^{\parallel}}{E_i^{\parallel}} = \frac{n_2 cos \theta_i - n_1 cos \theta_t} {n_2 cos \theta_i + n_1 cos \theta_t}
$$

# Appendix
Maxwell Equation:

$$
\begin{align*}
&\textbf{Differential Form} && \textbf{Integral Form} \\
&\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0} && \oint \mathbf{E} \cdot d\mathbf{a} = \frac{Q_{\text{enc}}}{\varepsilon_0} \\
&\nabla \cdot \mathbf{B} = 0 && \oint \mathbf{B} \cdot d\mathbf{a} = 0 \\
&\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} && \oint \mathbf{E} \cdot d\mathbf{l} = -\int \frac{\partial \mathbf{B}}{\partial t} \cdot d\mathbf{a} \\
&\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t} && \oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} + \mu_0 \varepsilon_0 \int \frac{\partial \mathbf{E}}{\partial t} \cdot d\mathbf{a} \\
\end{align*}
$$


# Ref:
https://youtu.be/gizFIrIVVPQ?si=Rc48WhKeLC-_nC4j
https://youtu.be/segy79MZlgM?si=LtyUAtlGg2b88lGt
