---
theme: base
paginate: true
header: "![w:50px](imgs/quantco.svg)"
footer: Kevin Klein [@kevkle](https://twitter.com/kevkle), Francesc Martí Escofet [@fmartiescofet](https://twitter.com/fmartiescofet)
---

![bg](imgs/monet-library.png)

<!-- _color: "white" -->
<!-- _footer: ''-->
<!-- _header: ''-->

# CATEring to Causal Inference: An Introduction to `metalearners`

## Flexible MetaLearners in Python

---

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 100%](imgs/monet-morpheus-wide.png)

---

<!--
'The unexamined life is not worth living' - Socrates (~400 BC)
'Ignorance is Bliss' - Thomas Gray (1742) / Publilius Syrus (43 BC)
-->

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 40%](imgs/fight.jpg)

---

<!--
'The unexamined life is not worth living' - Socrates (~400 BC)
'Ignorance is Bliss' - Thomas Gray (1742) / Publilius Syrus (43 BC)
-->

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 40%](imgs/fight_2.jpg)

---

## Let the data decide!

- We not only want to predict what happens in a world in which we don't have influence on the environment/data generating process.

- Rather, we want to **decide** which **intervention** to choose.

- Intuitively, in order to decide, we'd like to **compare the 'outcomes'** if taking the blue pill compared to the outcome of taking the red pill.
  $$Y(\text{blue pill}) - Y(\text{red pill})$$

---

## The data

- We have empirical **data** on individuals, who have

  - had their **properties**/covariates evaluated at the time of the intervention
  - been subject to the **intervention** (taking either of both pills)
  - had the **outcome** of their happiness measured, 5 years after the intervention

---

## The data, more formally

Our experiment data contains three different kinds of quantities per individual $i$:

| Name         | Symbol | Definition                                                                     |
| ------------ | ------ | ------------------------------------------------------------------------------ |
| Covariates   | $X_i$  | Properties of the time of intervention                                         |
| Intervention | $W_i$  | $=\begin{cases} 1 & \text{if red pill} \\ 0 & \text{if blue pill} \end{cases}$ |
| Outcome      | $Y_i$  | Happiness score ($\mathbb{R}$) 5 years after intervention                      |

$\mathcal{D} = \{ (X_i, W_i, Y_i)\}$

---

## $X$: Properties/covariates

|         | age | #phil books | dominant personality trait | education | atheist | ... |
| ------- | --- | ----------- | -------------------------- | --------- | ------- | --- |
| Anne    | 85  | 14          | extraversion               | BA        | 1       | ... |
| Bob     | 17  | 1           | openness                   | HS        | 0       | ... |
| Charlie | 34  | 0           | conscientiousness          | PhD       | 1       | ... |
| ...     | ... | ...         | ...                        | ...       | ...     | ... |

<!-- Big five personality traits:
Openness, coscientiousness, extraversion, agreeableness, neurotocism
-->

---

## Treatment effects

Many Causal Inference techniques allow for the estimation of the **Average Treatment Effect**

$$\mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill})]$$

---

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 100%](imgs/monet-snowflake.png)

---

![](imgs/ites_plain.png)

---

![](imgs/ites_classified.png)

---

## Capturing heterogeneity

Instead of estimating

$$\mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill})]$$

we'd like to estimate

$$Y_i(\text{blue pill}) - Y_i(\text{red pill})$$

---

## But how?

In an ideal world:

|         | age | #phil books | ... | $Y(\text{blue pill)}$ | $Y(\text{red pill})$ | $\tau$ |
| ------- | --- | ----------- | --- | --------------------- | -------------------- | ------ |
| Anne    | 85  | 14          | ... | 15.8                  | 13.2                 | 2.6    |
| Bob     | 17  | 1           | ... | 32.3                  | 32.4                 | -0.1   |
| Charlie | 34  | 0           | ... | 7.0                   | 9.2                  | -2.2   |
| ...     | ... | ...         | ... | ...                   | ...                  | ...    |

<p style="visibility:hidden"> The <strong>fundamental problem of Causal Inference</strong> states that we can never observe $Y(\text{blue pill})$ and $Y(\text{red pill})$ simultaneously for the same 'unit'. <p>

<!-- This would be just another regression problem -->

---

## But how?

In reality:

|         | age | #phil books | ... | $Y(\text{blue pill)}$             | $Y(\text{red pill})$              | $\tau$                            |
| ------- | --- | ----------- | --- | --------------------------------- | --------------------------------- | --------------------------------- |
| Anne    | 85  | 14          | ... | 15.8                              | <span style="color:red;">?</span> | <span style="color:red;">?</span> |
| Bob     | 17  | 1           | ... | <span style="color:red;">?</span> | 32.4                              | <span style="color:red;">?</span> |
| Charlie | 34  | 0           | ... | <span style="color:red;">?</span> | 9.2                               | <span style="color:red;">?</span> |
| ...     | ... | ...         | ... | ...                               | ...                               | ...                               |

The **fundamental problem of Causal Inference** states that we can never observe $Y(\text{blue pill})$ and $Y(\text{red pill})$ simultaneously for the same 'unit'.

<!-- This is not an off-the-shelf prediction problem -->

---

## What now?

- We can't know the Individual Treatment Effect (ITE)
  $$Y_i(\text{blue pill}) - Y_i(\text{red pill})$$
- Yet, we **can estimate** the **Conditional Average Treatment Effect (CATE)**
  $$\tau(X) := \mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill})|X]$$
  - Note the difference from the Average Treatment Effect
    $$\mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill})]$$

---

## Learning a policy

In order to get to a **decision** as to give what pill to whom, we can now follow the following process:

1. Estimate CATEs $\hat{\tau}(X) = \mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill}) | X]$
2. Distribute pills according to this policy:
   $\pi(X) := \begin{cases} \text{blue pill} & \text{if } \hat{\tau}(X) \geq 0 \\ \text{red pill} & \text{if }\hat{\tau}(X) < 0 \end{cases}$

---

## Estimating CATEs

- There are various approaches for estimating CATEs.
- MetaLearners are a family of approaches for estimating CATEs.
  - Work by [Chernozhukov(2016)](https://arxiv.org/abs/1608.00060), [Nie(2017)](https://arxiv.org/pdf/1712.04912), [Kunzel(2017)](https://arxiv.org/pdf/1706.03461), [Kennedy(2020)](https://arxiv.org/pdf/2004.14497) and more.

---

## MetaLearners

![bg right 50%](imgs/metalearner2.drawio.svg)

- MetaLearners are **CATE models** which rely on typical, **arbitrary machine learning estimators** (classifiers or regressors) as **components**.
- Some examples include the S-Learner, T-Learner, F-Learner, X-Learner, R-Learner, M-Learner and DR-Learner.

---

## MetaLearners

![bg right 80%](imgs/metalearner.drawio.svg)

- Input
  - $W$: Treatment assignments
  - $X$: Covariates/features
  - $Y$: Outcomes
- Output
  - $\hat{\tau}(X)$: CATE estimates

---

![width:400px](imgs/qr-metalearners.svg) ![width:400px](imgs/qr-presentation.svg)

[github.com/QuantCo/metalearners](https://github.com/QuantCo/metalearners)
[github.com/kklein/pdp24-metalearners](https://github.com/kklein/pdp24-metalearners)

---

# Backup

---

## Conventional assumptions for estimating CATEs

- Positivity/overlap
- Conditional ignorability/unconfoundedness
- Stable Unit Treatment Value (SUTVA)

A randomized control trial usually gives us the first two for free.

For more information see e.g. [Athey and Imbens, 2016](https://arxiv.org/pdf/1607.00698.pdf).

---

## Python implementations of MetaLearners

|                                           | `metalearners` | `causalml` | `econml` |
| ----------------------------------------- | :------------: | :--------: | :------: |
| MetaLearner implementations               |       ✔️       |     ✔️     |    ✔️    |
| Support\* for `pandas`, `scipy`, `polars` |       ✔️       |     ❌     |    ❌    |
| HPO integration                           |       ✔️       |     ❌     |    ❌    |
| Concurrency across base models            |       ✔️       |     ❌     |    ❌    |
| >2 treatment variants                     |       ✔️       |     ✔️     |    ❌    |
| Classification\*                          |       ✔️       |     ❌     |    ❌    |
| Other Causal Inference methods            |       ❌       |     ✔️     |    ✔️    |
