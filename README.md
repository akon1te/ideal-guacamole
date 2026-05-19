# ODE based exps for Master's degree paper


# Dynamical systems

## 1. Rossler system (1-year)

## 2. Quasi-periodic generator

$$
\dot{x} = y,
$$

$$
\dot{y} = (\lambda + z + x^2 - \beta x^4)\,y - \omega_0^2 x,
$$

$$
\dot{z} = -z - \kappa y^2.
$$

**Parameters (set 1):**
- $\beta=\frac{1}{18}$, $\kappa=0.02$, $\lambda=0.5$, $\omega_0=5.1$

**Parameters (set 2):**
- $\beta=\frac{1}{18}$, $\kappa=0.02$, $\lambda=1.0$, $\omega_0=5.1$

## 3. Hindmarsh–Rose system

$$
\dot{x} = y - a x^3 + b x^2 - z + I,
$$

$$
\dot{y} = c - d x^2 - y,
$$

$$
\dot{z} = r\bigl(s(x-\alpha) - z\bigr).
$$

**Parameters:**
- $a=1$, $b=3$, $c=1$, $d=5$, $s=4$, $\alpha=-1.6$

## 4. Sherman–Rinzel systems

$$
\tau \dot{V} = -I_{Ca}(V) - I_{K}(V,n) - I_{S}(V,S),
$$

$$
\tau_n \dot{n} = \sigma\bigl(n_\infty(V) - n\bigr),
$$

$$
\tau_s \dot{S} = S_\infty(V) - S.
$$

$$
I_{Ca}(V) = g_{Ca} m_\infty(V)(V - V_{Ca}),
$$

$$
I_{K}(V,n) = g_{K} n (V - V_{K}),
$$

$$
I_{S}(V,S) = g_{S} S (V - V_{K}).
$$

$$
f_\infty(V)=\left[1+\exp\left(\frac{V_f - V}{\theta_f}\right)\right]^{-1}.
$$

### Таблица 1. Параметры модели

| Parameter | Value | Parameter | Value | Parameter | Value |
|---|---:|---|---:|---|---:|
| $\tau$ | $0.02\,s$ | $\tau_s$ | $35\,s$ | $\sigma$ | $0.93$ |
| $g_{Ca}$ | $3.6$ | $g_{K}$ | $10.0$ | $g_{S}$ | $4.0$ |
| $V_{Ca}$ | $25.0\,mV$ | $V_{K}$ | $-75.0\,mV$ |  |  |
| $\theta_m$ | $12.0\,mV$ | $\theta_n$ | $5.6\,mV$ | $\theta_s$ | $10.0\,mV$ |
| $V_m$ | $-20.0\,mV$ | $V_n$ | $-16.0\,mV$ | $V_s$ | $-35.0\,mV$ |
