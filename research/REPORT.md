# Reinforcement learning in a live fighting game from pixels: a technical report

*brawl_scratch, November 2026. An agent that plays Brawlhalla (Wu Shang against a Mordex bot, Brawlhaven
small) from the screen and the keyboard only, in real time, on one laptop. This report states the problem
formally, explains with theory why each method failed or worked, and derives what should come next. Every
number comes from [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) and the files under [results/](../results/).*

---

## Contents

1. [The problem, formally](#1-the-problem-formally)
2. [Measurement: how small an effect can one laptop see?](#2-measurement-how-small-an-effect-can-one-laptop-see)
3. [Perception as state estimation](#3-perception-as-state-estimation)
4. [Why policy-gradient fine-tuning destroyed the clone](#4-why-policy-gradient-fine-tuning-destroyed-the-clone)
5. [Why greedy recurrent Q-learning failed three times](#5-why-greedy-recurrent-q-learning-failed-three-times)
6. [What worked: the advised clone as one KL-regularised improvement step](#6-what-worked-the-advised-clone-as-one-kl-regularised-improvement-step)
7. [Why it plateaus, and why distillation did not lift it](#7-why-it-plateaus-and-why-distillation-did-not-lift-it)
8. [Vision: information versus estimation](#8-vision-information-versus-estimation)
9. [Failure modes, options and the recovery reflex](#9-failure-modes-options-and-the-recovery-reflex)
10. [Latency and anticipation: the fighter-state network](#10-latency-and-anticipation-the-fighter-state-network)
11. [Generalising across bot levels](#11-generalising-across-bot-levels)
12. [How to make it stronger and more efficient](#12-how-to-make-it-stronger-and-more-efficient)
13. [Threats to validity](#13-threats-to-validity)
14. [Summary of results](#14-summary-of-results)
15. [References](#15-references)

---

## 1. The problem, formally

### 1.1 A continuing, real-time POMDP with no reset

The game is a partially observed Markov decision process $\mathcal{M}_d = (\mathcal{S}, \mathcal{A}, \Omega, P_d, O, r, \gamma)$
whose transition kernel $P_d$ contains the bot's policy $\mu_d$ at difficulty $d \in \{\text{easy}, \dots, \text{chosen}\}$:

$$
P_d(s' \mid s, a) = \sum_{b} \mu_d(b \mid s)\, P_{\text{game}}(s' \mid s, a, b).
$$

- **State** $s_t$: the game's memory, never read.
- **Observation** $o_t = h(\text{frame}_t)$: a vector of 112 measured quantities (positions, velocities, damage, weapons,
  stage geometry, and timers since measured events) produced by a detector, a tracker and a HUD reader.
- **Action** $a_t \in \mathcal{A} = \{9 \text{ directions}\} \times \{6 \text{ buttons}\}$, $|\mathcal{A}| = 54$, held for
  $k = 3$ game frames: one decision every $\Delta = 50$ ms (20 Hz).
- **Reward**, from HUD events only:
  $r_t = 0.01\,(\text{dealt}_t - \text{taken}_t) + (\text{KO}^{\text{scored}}_t - \text{KO}^{\text{lost}}_t)$,
  so 100 damage is worth one stock.

There is no episode boundary and no reset: the game cannot be put back. The task is continuing, and learning
uses the discounted return with $\gamma = 0.99$. Its effective horizon is $1/(1-\gamma) = 100$ decisions, and its half-life is
$\ln 2 / (1-\gamma) \approx 69$ decisions, or 3.5 s of play. The figure of merit is reward per minute of game time,
an estimate of the average reward $\rho^\pi = \lim_{T\to\infty} \frac{1}{T}\mathbb{E}_\pi\big[\sum_{t<T} r_t\big]$.

### 1.2 Real time means the world does not wait

Two properties make this setting unlike simulated benchmarks:

1. **Acting is delayed.** The action chosen from frame $t$ reaches the game after the capture, perception and
   decision latency $L$ (25 ms median here, 34 ms at p95, against a 50 ms budget). This is a *delayed MDP*
   (Katsikopoulos and Engelbrecht, 2003). With a constant delay it becomes Markov again on the augmented
   state $(s_t, a_{t-1})$, and the real-time RL framing of Ramstedt and Pal (2019) makes the same point. The
   timers and the GRU in the agent carry exactly that augmentation.
2. **Learning steals time.** A synchronous update freezes the agent while the opponent keeps playing (0.3 s per PPO
   update, measured). The final system therefore splits into an actor process that never waits and a learner
   process that trains from a queue. The stall's own damage is logged apart and is never credited to an action.

### 1.3 The sample budget

| Quantity | Value |
|---|---|
| Decisions per hour | 72,000 |
| Decisions per night (10 h) | 700,000 |
| Human demonstrations | 110 min against Medium (131,599 decisions; the human won 166–36 on stocks) |
| Environment instances | 1, no parallelism, no speed-up |

So one night is about $7\times10^5$ transitions, three orders of magnitude below a typical Atari agent's budget.
Every design decision below follows from that scarcity.

---

## 2. Measurement: how small an effect can one laptop see?

### 2.1 The noise floor

The same frozen policy (the human clone) was played for six consecutive 5-minute blocks (gate N1). Its reward per
minute had a block standard deviation of $\hat\sigma = 0.58$. For a two-sample comparison of $n$ blocks per arm,
with two-sided level $\alpha$ and power $1-\beta$, the minimum detectable effect is

$$
\delta_{\min} = (z_{1-\alpha/2} + z_{1-\beta})\,\hat\sigma\,\sqrt{2/n}.
$$

At $n = 6$ (30 minutes per arm), $\alpha = 0.05$ and power 0.8, $\delta_{\min} \approx 2.8 \times 0.58 \times 0.577 = 0.94$ reward per minute.
That is larger than every improvement this project has produced. **Half-hour comparisons can only detect huge
effects**, so the project moved to night-long designs.

### 2.2 Interleaved A/B nights

The laptop's speed drifts within a night: night 3 sensed 7 ms slower from its first minute, and 40% more frames
were late. If arm A runs in hour 1 and arm B in hour 2, that drift is confounded with the treatment. Interleaving
the two arms every 5 minutes and pairing each A block with the next B block gives

$$
\operatorname{Var}(\bar D) = \frac{\sigma_A^2 + \sigma_B^2 - 2\rho\,\sigma_A\sigma_B}{n},
$$

where $\rho > 0$ is the correlation that slow drift creates between neighbouring blocks. Pairing removes the
shared component, and the 5-minute alternation makes the drift common to both arms.

**Measured on night 4** (52 ten-minute cycles): the standard error of the paired difference was 0.10 reward per minute,
so $\delta_{\min} \approx 2.8 \times 0.10 = 0.28$. Night 5 used the same design and detected the reflex's harm at
$t = -4.1$ in a single night. The interleaved night is the project's real measuring instrument.

---

## 3. Perception as state estimation

The observation $o_t$ is an estimate of a few coordinates of $s_t$. The pipeline is:
DXGI capture → YOLO11s-pose (TensorRT FP16, about 14 ms) → a tracker that keeps identities and estimates
velocities → a HUD reader → a ledger that turns HUD changes into events.

**Gates passed on held-out frames:**
- fighters detected 98.6–100% of the time;
- identity swaps at most 0.7%;
- torso position error p95 at most 15 px at 1920×1080.

Two estimation problems deserve attention.

1. **The fall-versus-KO ledger is delayed.** Whether a lost stock was a KO or a self-inflicted fall is only known
   once the other fighter's "+2" badge shows or fails to show, $C = 240$ frames (80 decisions) later. The penalty
   itself is immediate (the stock counter drops), so the reward is unaffected. Analyses of falls, however, must
   map each `fell` event back to the preceding stock loss. Doing this naively misattributed every fall to the wrong
   moment.
2. **Off-screen states are censored.** Beyond the screen's edge the fighter is undetected. The observation keeps
   the last seen position with zero velocity. On night 4 this happened 483 times while off stage, about 1.8 s each,
   and 27 of those spells ended in a lost stock. Any rule that requires the fighter to be "present" is blind exactly
   when recovery matters most.

---

## 4. Why policy-gradient fine-tuning destroyed the clone

**Result.** Over two independent eras, PPO started from the behaviour-cloned policy made it worse:
- Round A, September: −1.2 net stocks per minute after 50k steps.
- Stage 1, November: −0.57 reward per minute within an hour, 2.1 standard errors, Mann–Whitney $p = 0.012$, with no
  recovery.

Three mechanisms explain it.

### 4.1 The gradient's signal-to-noise ratio

PPO estimates $\nabla J = \mathbb{E}\big[\nabla\log\pi(a\mid s)\,\hat A(s,a)\big]$ from one rollout of $N = 1{,}200$ decisions. The decisions
are strongly autocorrelated: a fighter's situation persists over many decisions. The effective sample size is

$$
N_{\text{eff}} = \frac{N}{1 + 2\sum_{k\ge1} \rho_k},
$$

which is far smaller than $N$ when the autocorrelations $\rho_k$ of the summand decay slowly. The gradient's signal-to-noise ratio scales as
$\lVert \nabla J\rVert\sqrt{N_{\text{eff}}}/\sigma$. With a reward whose 5-minute block standard deviation is as large as the
effects being sought (Section 2), each update is mostly noise.

### 4.2 An untrained critic

The cloned weights give a policy but no value function. The first advantages are
$\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, computed from a random $V$. The first
updates therefore follow noise, and they act on the very policy one wants to keep.

### 4.3 A cloned policy lives on a narrow manifold

The performance difference lemma (Kakade and Langford, 2002) states

$$
J(\pi') - J(\pi) = \frac{1}{1-\gamma}\,\mathbb{E}_{s\sim d^{\pi'}}\,\mathbb{E}_{a\sim\pi'}\big[A^\pi(s,a)\big].
$$

A random perturbation of $\pi$ has an expected advantage near zero on states where $\pi$ is already sensible. But
it moves the state distribution $d^{\pi'}$ toward states the human never visited, where the clone has no
competence. The compounding-error analysis of imitation (Ross, Gordon and Bagnell, 2011) bounds the clone's cost by
$O(\varepsilon T^2)$ for per-step error $\varepsilon$ over horizon $T$, which makes the clone *brittle*: a small change in
action probabilities compounds into a large change in where the agent goes. PPO's ratio clipping limits each
sample's ratio, not this cumulative drift.

**Lesson.** With one slow environment, unconstrained policy-gradient fine-tuning cannot beat the noise before it erodes
the prior. Whatever works has to keep the policy close to the clone by construction.

---

## 5. Why greedy recurrent Q-learning failed three times

The second engine was an R2D2-style recurrent Q-network (Kapturowski et al., 2019) learning from demonstrations
(DQfD, Hester et al., 2018). It had a GRU-256, a dueling head, double Q, 5-step returns and a margin loss on the
human's action. It failed in three distinct ways, each a known pathology.

### 5.1 Causal confusion: the copycat

With the last four actions in its input, the pretrained network chose the history's last action 88–90% of the
time, against 63–69% for the human. The previous action is a near-perfect predictor of the next human action, so
the network learned the shortcut $a_t \approx a_{t-1}$ instead of its causes. That is *causal confusion* (de Haan,
Jayaraman and Levine, 2019). Acting greedily closed the loop: it repeated its own action until it walked off the
stage, losing 0–18 stocks in 5 minutes. **Fix:** the history was removed from the input, and the copy rate fell to 0.16.

### 5.2 Overestimation under distribution shift

Overnight, the values inflated. The agent's own play, whose true return was about −0.4, was valued at about +1.1,
and the reward fell from −1.85 to −4.32 per minute. Two terms compound.

1. **The max operator** is biased upward under estimation noise (Thrun and Schwartz, 1993):

   $$
   \mathbb{E}\big[\max_a \hat Q(s,a)\big] \ge \max_a \mathbb{E}\big[\hat Q(s,a)\big].
   $$

   Double Q-learning (van Hasselt, 2010) reduces this bias but does not remove it.
2. **Extrapolation.** Bootstrapped targets query $\hat Q(s', a')$ at actions and states the data never covered, the
   central difficulty of offline RL (Fujimoto et al., 2019; Kumar et al., 2020). With function approximation,
   bootstrapping and off-policy data together (the "deadly triad"), nothing anchors those values.

### 5.3 Mode collapse of the margin loss

The DQfD margin loss

$$
\mathcal{L}_{\text{margin}} = \max_a\big[Q(s,a) + \ell(a_E, a)\big] - Q(s, a_E)
$$

pushes the human's action $a_E$ to the top. In ambiguous states the human's commonest action, "no button" (70% of
human decisions), therefore becomes the argmax. With exploration off, the greedy network stood still. **The
Q-network alone is the wrong player.**

### 5.4 The offline value benchmark

Before changing the online system again, value learning was measured offline. Each variant was trained on 10 human
sessions and scored on 5 held-out sessions, by how well $Q(s, a_{\text{human}})$ tracked the true discounted return.

| Variant | Correlation with the return |
|---|---|
| Regressing the true return directly | 0.34–0.39 |
| The night's settings: double Q, $\gamma$ 0.995, $n$ 5, target every 2,000, lr $10^{-4}$ | 0.25 |
| Best: $\gamma$ 0.99, $n$ 20, target every 500, lr $3\times10^{-5}$ | **0.43**; lose AUC 0.74, win AUC 0.83 |

The bias–variance trade-off of $n$-step returns explains the ranking. A larger $n$ reduces the weight of the
bootstrapped (biased) term, $\gamma^n \hat Q(s_{t+n}, \cdot)$, at the price of variance. Here $\gamma^{20} = 0.82$ against
$\gamma^{5} = 0.95$, and a shorter horizon ($\gamma = 0.99$) lowers the variance of every target. This benchmark became the gate
for every later value change.

---

## 6. What worked: the advised clone as one KL-regularised improvement step

### 6.1 The policy

The agent samples from the clone $\pi_0$, re-weighted by a recurrent Q-network's clipped advantages:

$$
\pi_\beta(a \mid s) = \frac{\pi_0(a\mid s)\,\exp\!\big(\beta \hat A(s,a)\big)}{Z_\beta(s)},
\qquad
\hat A(s,a) = \operatorname{clip}\!\Big(\hat Q(s,a) - \sum_b \pi_0(b\mid s)\hat Q(s,b),\, -c,\, c\Big),
$$

with $\beta = 10$ and $c = 0.1$. The Q-network is trained by 20-step **SARSA**: its target uses the action actually
taken, so there is no max and no overestimation. It therefore evaluates the policy that is playing, $Q^{\pi_\beta}$.

### 6.2 Why this form: the KL-regularised improvement step

$\pi_\beta$ is the closed-form solution of a trust-region problem. For each state, maximise the expected advantage under a
KL penalty toward the clone:

$$
\max_{\pi(\cdot\mid s)}\; \mathbb{E}_{a\sim\pi}\big[\hat A(s,a)\big] - \frac{1}{\beta}\,\mathrm{KL}\big(\pi(\cdot\mid s)\,\Vert\,\pi_0(\cdot\mid s)\big).
$$

Setting the derivative of the Lagrangian (with the normalisation constraint) to zero gives
$\log\pi(a\mid s) = \log\pi_0(a\mid s) + \beta\hat A(s,a) - \log Z_\beta(s)$, which is the policy above. The same step underlies
REPS (Peters et al., 2010), MPO (Abdolmaleki et al., 2018), AWR (Peng et al., 2019) and AWAC (Nair et al., 2020).
Here it is applied **at sampling time**, without a parametric policy update, so the clone itself never changes.

### 6.3 The step is bounded

Because $|\beta\hat A| \le \beta c = 1$ and $Z_\beta \in [e^{-\beta c}, e^{\beta c}]$, every probability ratio is bounded:

$$
e^{-2\beta c} \le \frac{\pi_\beta(a\mid s)}{\pi_0(a\mid s)} \le e^{2\beta c},
\qquad
\mathrm{KL}(\pi_\beta\Vert\pi_0) = \beta\,\mathbb{E}_{\pi_\beta}[\hat A] - \log Z_\beta \le 2\beta c,
$$

and by Pinsker's inequality $\mathrm{TV}(\pi_\beta, \pi_0) \le \sqrt{\mathrm{KL}/2}$. For small $\beta$ the KL is second order:
$\mathrm{KL}(\pi_\beta\Vert\pi_0) = \tfrac{\beta^2}{2}\operatorname{Var}_{\pi_0}(\hat A) + O(\beta^3)$.

On human play the advantages are small (median $|\hat A| = 0.04$, 90th percentile 0.12). At $\beta = 10$ the clone's most
likely action changes in only 5% of decisions.

This boundedness is exactly what PPO lacked. The agent stays on the support of the human's behaviour, where the
Q-network has data, so the extrapolation problem of Section 5.2 never arises.

### 6.4 How much a step can gain

Differentiate the expected action-value under $\pi_\beta$ with respect to $\beta$:

$$
\frac{d}{d\beta}\,\mathbb{E}_{a\sim\pi_\beta}\big[Q(s,a)\big]
= \mathbb{E}_{\pi_\beta}\big[Q\,(\hat A - \mathbb{E}_{\pi_\beta}\hat A)\big]
= \operatorname{Cov}_{a\sim\pi_\beta}\big(Q(s,a),\, \hat A(s,a)\big).
$$

Combined with the performance difference lemma, the first-order gain of the advised policy over the clone is

$$
J(\pi_\beta) - J(\pi_0) \approx \frac{\beta}{1-\gamma}\;\mathbb{E}_{s\sim d^{\pi_0}}\Big[\operatorname{Cov}_{a\sim\pi_0}\big(A^{\pi_0}(s,a),\, \hat A(s,a)\big)\Big] + O(\beta^2).
$$

The gain is proportional to the **within-state covariance between the true advantages and the estimated ones**. It is
not about how well $\hat Q$ ranks states. Writing $\operatorname{Cov} = \rho_s\,\sigma_{A}(s)\,\sigma_{\hat A}(s)$, three things limit the improvement:
- how much the actions differ in true value ($\sigma_A$);
- how much the advisor separates them ($\sigma_{\hat A}$, capped by $c$);
- above all, the **within-state correlation** $\rho_s$ of the advisor.

### 6.5 Result

| Measurement | Clone | Advised clone |
|---|---|---|
| Night 1, in-run (7 h; clone = first 30 min) | −0.11 ± 0.19 | **+0.38 ± 0.06** per minute (2.5 SE, Mann–Whitney $p = 0.016$) |
| Paired frozen evaluation (6 × 5 min each) | −0.08 ± 0.32; stocks 15–16; matches 4–3 | **+0.50 ± 0.21**; stocks 20–14; matches 6–1 |
| Night 4 (10 h, against Medium) | | stocks **399–386**, ahead on damage, +0.28 per minute |

This was the first method in the project's history to beat its own prior.

---

## 7. Why it plateaus, and why distillation did not lift it

### 7.1 One night is one improvement step

Every advised night levels off around +0.3 reward per minute above the clone. Night 2 continued from night 1's
Q-network with $\beta = 15$ and reached +0.26 against night 1's +0.38. This is what Section 6 predicts:

- **The base never moves.** $\pi_0$ is fixed, so the night performs *one* step of soft policy iteration,
  $\pi_0 \to \pi_\beta$, and then only refines $\hat Q$ for that same $\pi_\beta$. Soft policy iteration (as in SAC, Haarnoja
  et al., 2018) gains by repeating the step, $\pi_{k+1} \propto \pi_k \exp(\beta \hat A^{\pi_k})$, and the system never repeats it.
- **A larger $\beta$ does not help.** Beyond the linear regime the clip saturates. The gain stops growing like $\beta$,
  while the noise in $\hat A$ (the part not covariant with $A$) is amplified as much as the signal.
- **The advisor's quality caps the gain.** Its *across*-state value correlation is only 0.43–0.46. The *within*-state
  correlation $\rho_s$, which is what matters, has never been measured; Section 12.2 shows how.

### 7.2 Distillation: a signal smaller than the imitation error

The obvious way to move the base is to clone the advised play into a new $\pi_0'$ and advise that. On the night-4
A/B night, the distilled clone was **no better**: −0.09 ± 0.10 per minute paired over 52 cycles. The bound above
explains why. The advised policy differs from the clone by

$$
\mathrm{KL}(\pi_\beta\Vert\pi_0) \approx \tfrac{\beta^2}{2}\operatorname{Var}_{\pi_0}(\hat A),
$$

which is of order $10^{-1}$ nats per decision at the measured advantage scale. Behaviour cloning, by contrast, reaches a
held-out cross-entropy of 0.89 nats, $H(\pi_\beta) + \mathrm{KL}(\pi_\beta\Vert\pi_0')$. Unless the distillation's own KL term is
much smaller than $\mathrm{KL}(\pi_\beta\Vert\pi_0)$, the new base is as far from $\pi_\beta$ as the old one, in a random
direction, and the improvement is lost.

The distillation also used **hard labels**: sampled actions, kept only from stretches that ended in a stock taken.
That adds sampling variance and a selection bias on top. Yet $\pi_\beta$ is known in closed form at every visited state.
Section 12.1 turns this into a concrete fix.

---

## 8. Vision: information versus estimation

**The puzzle.** Adding both fighters' crops to the vector made value estimates *worse* on human data: lose AUC 0.61
against 0.73, worse in every fold.

**Why it is not about information.** By the data-processing inequality, the extra input can only add information:

$$
I\big(Y;\, (o^{\text{vec}}, o^{\text{img}})\big) \ge I\big(Y;\, o^{\text{vec}}\big).
$$

The loss is therefore estimation error. A CNN on 2.5 hours of data has far more effective capacity than signal, so
its excess risk (of order $\sqrt{\text{complexity}/N}$) exceeds the conditional information it can extract.

**The scaling test confirms it.** The same test was rerun on night 4's 700,000 decisions of the agent's own play,
280 times the human data, on Kaggle.

| Input | Correlation | Lose AUC | Win AUC |
|---|---|---|---|
| Vector | 0.442 | 0.771 | 0.846 |
| Vector + crops (lr $3\times10^{-5}$) | 0.457 (+0.015 ± 0.003) | 0.771 | 0.844 |

With enough data the pictures stop hurting. They add a small but consistent amount of correlation in both folds, and
they do not anticipate stocks any earlier. **For the value target**, $I(Y;\, o^{\text{img}} \mid o^{\text{vec}})$ is small,
because the vector already carries positions, damage and timers.

**The crops do carry action state.** A probe that read Wu Shang's action state, labelled for free from the key log,
showed that a single 96 px crop adds AUC +0.17 for "attacking" and +0.08 for "dodging" over kinematics alone.

**Reconciling the two.** The TD target is a weak, delayed and noisy signal. It does not supply enough gradient to learn
an "attack incoming" feature, even though that feature is in the pixels. **Supervision supplies it directly**, which
is the rationale for the fighter-state network (Section 10).

---

## 9. Failure modes, options and the recovery reflex

**Night 4's fall analysis** (against Medium):
- **35% of lost stocks** (137 of 386) were self-inflicted falls, at low damage (median 30%, against 71% for stocks lost to
  hits).
- Off stage the agent steered toward the stage on 66% of decisions, but used the recovery (up + heavy) on only 7.5%.
- It came back from 97% of off-stage spells; the 3% that failed cost a third of its stocks.

**Why RL does not fix this.** Failures are rare and delayed, so the value learning gets almost no signal. Credit must
travel back about 60 decisions through a discount of $\gamma^{60} = 0.55$, over a few hundred events per night, through an
advisor with a value correlation of 0.44.

**The fix: an option.** A hand-written rule, taking over only in a region of the state space, is an *option*
$\omega = (\mathcal{I}, \pi_\omega, \beta_\omega)$ (Sutton, Precup and Singh, 1999):
- the initiation set $\mathcal{I}$ = {beyond a ledge and falling} ∪ {below the stage's top};
- $\pi_\omega$ = toward the stage, then jumps, then up + heavy, then a dodge;
- termination $\beta_\omega$ = leaving $\mathcal{I}$.

The Q-network keeps learning by SARSA on the actions actually taken, so it evaluates the composite policy.

**Night 5 (against Hard, reflex off and on every 5 minutes):**
- The first version **doubled** the self-falls: 0.35 against 0.14 per minute.
- Paired difference over 52 cycles: −0.37 ± 0.09 reward per minute ($t = -4.1$).
- The traces showed why. Below the stage's top, beside the ledge, the rule rose straight up or steered away from the
  stage. The clone held toward the stage and clung to its side, which is how it came back from 97% of spells. With
  jumps, recovery and dodge spent, the rule's fighter fell straight down past the wall.

**Lessons with general reach:**
1. **An offline replay of a rule can measure where it would act, never what would follow.** Without a simulator the
   counterfactual is unobservable, so only the interleaved A/B measures an option's worth.
2. **The worth of fixing a failure mode depends on the opponent.** Self-falls were 35% of losses against Medium and 10%
   against Hard. The state distribution $d^\pi$ is opponent-dependent, and so is the marginal value of every skill.
3. **Corrected rule:** always hold toward the stage. It is being tested against Medium on night 6.

---

## 10. Latency and anticipation: the fighter-state network

### 10.1 Why reaction alone cannot defend

Let $L$ be the agent's total reaction latency: capture + perception (25 ms) + up to one decision period until the
next decision + the action period itself. Then $L \approx 50$–$100$ ms before an action takes effect. An attack whose
wind-up lasts $\tau_{\text{startup}}$ can be answered reactively only if $L + \tau_{\text{dodge}} < \tau_{\text{startup}}$.

Against Hard, the agent took 279 damage per minute while dealing 45, a 6:1 ratio. When the condition fails, the only
defence is **prediction**: acting on $p(\text{attack at } t + h \mid o_{\le t})$ with $h \approx L$.

### 10.2 The network

**Architecture:**
- a small CNN embeds each 96 px grey crop of Mordex into a token;
- a 2-layer transformer runs over the tokens of the newest frame and the five before it;
- the newest token feeds two heads:
  - $p(z_t \mid o_{t-5:t})$, the state now;
  - $p(z_{t+2} \mid o_{t-7:t-2})$, the state 100 ms ahead. In deployment this head runs on the latest frames.
- $z \in$ {neutral, attacking, dodging, recovering, hurt}.

Only past frames enter the network. The future is used only to build training targets. The agent will receive both
*distributions*, not their argmax, so the policy can weigh uncertainty. The design costs about 1 ms, because earlier
frames' tokens are cached.

### 10.3 Labels: the human, and free targets

**Hand labels.** About 1,500 so far, on clips drawn from nights 4 and 5, with over-sampling of the half second before
Wu Shang is hit. The label is the state *in the framed frame*. History is left to the network and the agent's memory.

**Free targets for pretraining** come from the night traces. They act as auxiliary tasks, in the manner of UNREAL
(Jaderberg et al., 2017):
- from Wu Shang's crops: the decisions since each button press and since the last hit, in 6 bins each;
- from Mordex's crops: whether Wu Shang is hit within 0.5 s (an attack is coming), and the decisions since Mordex was
  last hit.

**Why the free targets matter.** There are millions of free examples and very few hand labels for rare states (only
17 "dodging" labels). The supervised gradient for "what does an attack look like" arrives without waiting for TD to
discover it (Section 8).

**Evaluation.** Four variants (frame only or with context, each with or without pretraining) are compared on folds held
out by time. This isolates what context and pretraining each contribute before anything enters the live agent.

---

## 11. Generalising across bot levels

### 11.1 The problem

Training against one bot means optimising $J(\pi; \mu_{\text{medium}})$. The agent learned to exploit Medium's habits:
run in, light attack, then down-light or heavy. Hard punishes that habit. Formally, a policy that is optimal for
$P_{\text{medium}}$ can be arbitrarily bad for $P_{\text{hard}}$ once it relies on regularities of $\mu_{\text{medium}}$ that $\mu_{\text{hard}}$
does not share. Ghosh et al. (2021) show that generalisation from limited training contexts is itself partially
observed (an "epistemic POMDP"). The optimal generalising policy is generally *not* the optimal policy of any single
context.

### 11.2 Three principled levers

1. **A distribution over opponents, with the top levels held out.** Train on the mixture
   $\max_\pi \mathbb{E}_{d\sim p_{\text{train}}} J(\pi;\mu_d)$ with $p_{\text{train}}$ over {easy, medium, hard}. A recurrent policy can then
   infer $d$ from history (implicit system identification, as in domain randomisation, Tobin et al., 2017), and it
   approaches the Bayes-optimal policy for that prior.
   - Evaluate on {extreme, chosen}, never trained on, with learning frozen.
   - The game's level switch (Tab+D / Tab+A) is already coded.
   - Extrapolation beyond the training support carries no guarantee: Chosen is faster than anything in it, so the gap
     must be measured rather than assumed.
2. **Invariant features: respond to the opponent's state, not to his habits.** Mordex's moves and their frame data are
   the same at every level; only the choice distribution $\mu_d(b\mid s)$ changes. A response conditioned on the
   opponent's *observable* state $z_t$ ("he is in wind-up: dodge"; "he is in end lag: punish") has an approximately
   level-invariant value:

   $$
   Q^{*}_d(s, z, a) \approx Q^{*}(s, z, a) \quad \text{for reactive } a.
   $$

   A habit, by contrast, exploits $\mu_d(b\mid s)$ itself. This is the logic of invariant prediction: features whose
   relation to the outcome is stable across environments transfer. The fighter-state network supplies $z_t$.
3. **Pessimism as robustness.** A stronger opponent is, for the agent, an adversarial perturbation of the transition
   kernel. Optimising the lower tail of the return is equivalent to robustness against such perturbations: CVaR at
   level $\alpha$ equals the worst-case expectation over transition kernels whose density ratio to the nominal one is at
   most $1/\alpha$ (Chow et al., 2015). A distributional advisor (quantile regression, Dabney et al., 2018) that ranks
   actions by a lower quantile would discount moves that work only because Medium punishes late.

---

## 12. How to make it stronger and more efficient

Ordered by expected gain per hour of work.

### 12.1 Iterate the improvement step offline, between nights

**Combine Sections 6 and 7 into a day–night loop:**
1. Nights collect data with the advised policy (about 3.5M decisions are already traced).
2. Days run several soft policy-iteration steps on Kaggle, entirely offline.
3. Each step evaluates with in-sample methods that never query unseen actions: SARSA, or IQL's expectile regression
   (Kostrikov et al., 2021). Then it extracts the new base by **soft-target distillation**:

   $$
   \pi_{k+1} = \arg\min_\pi \; \mathbb{E}_{s\sim\mathcal{D}}\Big[\mathrm{KL}\big(\pi_k(\cdot\mid s)\,e^{\beta\hat A_k(s,\cdot)}/Z \;\Vert\; \pi(\cdot\mid s)\big)\Big].
   $$

**Why soft targets.** The target distribution is computed exactly at every visited state, so there is no sampling
variance and no selection bias. Distillation with soft targets (Hinton et al., 2015; Rusu et al., 2016) can therefore
preserve a tilt of 0.1 nats, which hard labels lost. The KL bound per step keeps each iterate on the data's support,
and $K$ steps per day replace the single step per night.

**A direct diagnostic, cheap to run on the existing traces:** compare $\mathbb{E}_s\mathrm{KL}(\pi_\beta\Vert\pi_0')$ with
$\mathbb{E}_s\mathrm{KL}(\pi_\beta\Vert\pi_0)$. The distillation has worked only if the first is much smaller.

### 12.2 Measure what the gain depends on: within-state advantage accuracy

Section 6.4 says the gain scales with the within-state correlation $\rho_s$, but the benchmark measures across-state
value correlation. **A within-state estimate** can be built from logged data:
1. bin similar states (a k-NN over the observation);
2. within each bin, compare the realised $n$-step returns of different actions;
3. correlate those contrasts with $\hat A$.

This becomes the new offline gate for advisor changes: dueling-head variants, $\lambda$-returns, Retrace for
off-policy correctness (Munos et al., 2016), and value rescaling.

### 12.3 Shape recovery with a potential, not a rule

Night 5 showed that a rule can be wrong. **Potential-based shaping** (Ng, Harada and Russell, 1999),

$$
F(s, s') = \gamma\,\Phi(s') - \Phi(s),
$$

leaves the optimal policy unchanged for any potential $\Phi$, while densifying the delayed fall penalty. A natural potential
is a learned or hand-set *recoverability*:
$\Phi(s) = -\kappa \cdot \mathbb{1}[\text{off stage}] \cdot g(\text{distance beyond the ledge}, \text{height below the top}, \text{jumps spent})$.
Credit for staying recoverable then arrives at every decision, instead of 60–80 decisions later through $\gamma^{60}$.

### 12.4 Self-imitation of the agent's own wins

The clone copies only the human's winning stretches. Self-imitation learning (Oh et al., 2018) adds the agent's own
transitions whose return exceeds the value estimate:
$\mathcal{L}_{\text{SIL}} = \mathbb{E}\big[-\log\pi(a\mid s)\,(R - V(s))_+\big]$.
This is an online, principled version of "keep what won", and it moves the base toward the agent's own successes.

### 12.5 Spend the compute where the data is

- **Replay ratio.** The learner runs at 0.4 updates per decision on the CPU, because the detector keeps the 4 GB
  GPU. Offline days (Section 12.1) remove that constraint.
- **Latency.** INT8 TensorRT, or a smaller input, reduces $L$ in Section 10.1. Every millisecond saved counts directly
  against the opponent's attack startup.
- **Evaluation.** Variance can be cut further with control variates, in the style of CUPED. Regressing each block's
  reward on pre-treatment covariates, such as the clone's own score in the neighbouring blocks, reduces the variance
  by $(1-R^2)$ at no cost in bias.

### 12.6 A curriculum over opponents

Rotate {easy, medium, hard} within nights, keep {extreme, chosen} for frozen tests (Section 11), and add the opponent's
state to the observation. Human demonstrations against Hard (being recorded) give a prior that has seen Hard's speed.

---

## 13. Threats to validity

- **One seed per configuration.** The nights are long, but each is one run. The interleaved design controls drift
  within a night, not run-to-run variance.
- **Short in-run baselines.** The clone-only baselines are 30 minutes (standard error about 0.2), shorter than the
  advised segments. The paired frozen evaluation and the A/B nights address this.
- **Sensing errors become reward errors.** About 30 of night 4's 137 "falls" showed Wu Shang standing on the stage just
  before the stock was lost. These are probably KOs whose "+2" credit was misread. Fall statistics are therefore
  upper bounds.
- **One matchup, one stage, one laptop.** Every claim is about Wu Shang against Mordex on Brawlhaven small.

---

## 14. Summary of results

| Method | Against | Outcome |
|---|---|---|
| Goal-conditioned PPO/SAC, skill ladders (2026-03 to 09) | Easy, Medium | Skill collapse, "always jump" exploit; net stocks negative at every evaluation |
| Behaviour cloning, 110 min, winning stretches (medium_v4) | Medium | Even: −0.03 ± 0.24 per minute, stocks 16–17 |
| PPO from the clone | Medium | **Worse**: −0.57 per minute within an hour ($p = 0.012$) |
| Greedy recurrent Q-learning | Medium | **Worse**: copycat, then inflation (−1.85 → −4.32), then standing still |
| **Advised clone** (KL-bounded sampling, SARSA advisor) | Medium | **+0.38 ± 0.06** in-run against −0.11; frozen +0.50 against −0.08; even on stocks over 10 h |
| Distilled base (hard labels) | Medium | No better: −0.09 ± 0.10 paired |
| Crops for the value (2.5 h of data, then 700k decisions) | Medium | Worse (lose AUC −0.12), then neutral (+0.015 correlation) |
| Advised clone | Hard | Outclassed: stocks 66–409, damage 45 against 279 per minute |
| Recovery reflex v1 | Hard | **Harmful**: self-falls ×2.4, $t = -4.1$; corrected version on test |

**What the project established:**
1. A real-time, single-instance, pixels-to-keys RL testbed with measured perception (gates), timing (25 ms per decision)
   and noise (Section 2).
2. In this regime, unconstrained fine-tuning of a cloned policy fails, with policy gradients and with greedy Q-learning,
   for identifiable reasons.
3. A KL-bounded, advantage-tilted clone improves reliably. Its gain and its plateau follow from the first-order theory
   of Section 6.4.
4. Pixels did not help the value at either data scale tested: they hurt it at 2.5 hours and were neutral at 700,000
   decisions. They do carry the opponent's state, which needs supervision to extract.
5. Interleaved A/B nights are a measuring instrument for effects down to about 0.3 reward per minute.

---

## 15. References

- Abdolmaleki, A. et al. (2018). Maximum a posteriori policy optimisation. *ICLR*.
- Chow, Y., Tamar, A., Mannor, S., Pavone, M. (2015). Risk-sensitive and robust decision-making: a CVaR optimization approach. *NeurIPS*.
- Dabney, W., Rowland, M., Bellemare, M., Munos, R. (2018). Distributional reinforcement learning with quantile regression. *AAAI*.
- de Haan, P., Jayaraman, D., Levine, S. (2019). Causal confusion in imitation learning. *NeurIPS*.
- Fujimoto, S., Meger, D., Precup, D. (2019). Off-policy deep reinforcement learning without exploration. *ICML*.
- Ghosh, D. et al. (2021). Why generalization in RL is difficult: epistemic POMDPs and implicit partial observability. *NeurIPS*.
- Haarnoja, T. et al. (2018). Soft actor-critic. *ICML*.
- Hester, T. et al. (2018). Deep Q-learning from demonstrations. *AAAI*.
- Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Jaderberg, M. et al. (2017). Reinforcement learning with unsupervised auxiliary tasks. *ICLR*.
- Kakade, S., Langford, J. (2002). Approximately optimal approximate reinforcement learning. *ICML*.
- Kapturowski, S. et al. (2019). Recurrent experience replay in distributed reinforcement learning. *ICLR*.
- Katsikopoulos, K., Engelbrecht, S. (2003). Markov decision processes with delays and asynchronous cost collection. *IEEE TAC*.
- Kostrikov, I., Nair, A., Levine, S. (2021). Offline reinforcement learning with implicit Q-learning. *ICLR 2022*.
- Kumar, A., Zhou, A., Tucker, G., Levine, S. (2020). Conservative Q-learning for offline RL. *NeurIPS*.
- Munos, R., Stepleton, T., Harutyunyan, A., Bellemare, M. (2016). Safe and efficient off-policy reinforcement learning. *NeurIPS*.
- Nair, A., Gupta, A., Dalal, M., Levine, S. (2020). AWAC: accelerating online RL with offline datasets. *arXiv:2006.09359*.
- Ng, A., Harada, D., Russell, S. (1999). Policy invariance under reward transformations. *ICML*.
- Oh, J., Guo, Y., Singh, S., Lee, H. (2018). Self-imitation learning. *ICML*.
- Peng, X. B., Kumar, A., Zhang, G., Levine, S. (2019). Advantage-weighted regression. *arXiv:1910.00177*.
- Peters, J., Mülling, K., Altun, Y. (2010). Relative entropy policy search. *AAAI*.
- Ramstedt, S., Pal, C. (2019). Real-time reinforcement learning. *NeurIPS*.
- Ross, S., Gordon, G., Bagnell, J. A. (2011). A reduction of imitation learning to no-regret online learning. *AISTATS*.
- Rusu, A. et al. (2016). Policy distillation. *ICLR*.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Sutton, R., Precup, D., Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal abstraction. *Artificial Intelligence*.
- Thrun, S., Schwartz, A. (1993). Issues in using function approximation for reinforcement learning. *Connectionist Models Summer School*.
- Tobin, J. et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*.
- van Hasselt, H. (2010). Double Q-learning. *NeurIPS*.
