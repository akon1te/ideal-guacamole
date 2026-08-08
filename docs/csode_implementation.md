# ControlSynthODE baseline

The public `csode` model follows the executable main-experiment form in the
official ControlSynth Neural ODE repository, not the former dissipative model
that previously used this name in this project.

For an autonomous observed state `x`, it integrates a joint state `[x, u]`:

\[
\dot{x}=f_\theta(t,x)+g_\phi(t,u),\qquad
\dot{u}=g_\phi(t,u),\qquad u_0=x_0.
\]

`f_theta` is the main time-conditioned Tanh MLP. `g_phi` is the auxiliary,
time-conditioned control network; a one-layer linear control network is the
default, matching the authors' `ODEFuncG`. The same `g_phi(t, u)` is both the
control state derivative and the term added to the physical state derivative.

The original paper uses Euler in its main experiments. This project defaults
to RK4 so all ODE baselines share a solver; pass `--csode_solver euler` for a
solver-level reproduction experiment. The paper's LMI certificate is outside
the scope of this baseline and must not be claimed from this architecture
alone.

Relevant options:

```text
--csode_layers 3
--csode_control_hidden <defaults to --hidden>
--csode_control_layers 1
--csode_solver {rk4,euler}
```
