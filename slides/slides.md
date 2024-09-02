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

# CATEring Causal Inference: An Introduction to `metalearners`

## Flexible MetaLearners in Python

---

## Agenda

- Why care about treatment effect estimation? (5')
- Introduction to MetaLearners for treatment effect estimation (5')
- Some shortcomings with existing libraries (5')
- Demo (10')
- Q&A (5')

---

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 97%](imgs/monet-morpheus-wide.png)

---

<!--
'The unexamined life is not worth living' - Socrates (~400 BC)
'Ignorance is Bliss' - Thomas Gray (1742) / Publilius Syrus (43 BC)
-->

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 40%](imgs/argument.png)

---

## Let the data decide!

Let's learn from data which pill we should take.

Empirical **data** on individuals, who have

- had their **properties**/covariates evaluated at the time of the intervention
- been subject to the **intervention** (taking either of both pills)
- had the **outcome** of their happiness measured, 5 years after the intervention

---

## Data setup

Our experiment data contains three different kinds of quantities:

| Name         | Symbol | Definition                                                                     |
| ------------ | ------ | ------------------------------------------------------------------------------ |
| Covariates   | $X_i$  | Properties of the time of intervention                                         |
| Intervention | $W_i$  | $=\begin{cases} 1 & \text{if red pill} \\ 0 & \text{if blue pill} \end{cases}$ |
| Outcome      | $Y_i$  | Happiness score ($\mathbb{R}$) 5 years after intervention                      |

---

## Treatment effects

As a first step, we can try to estimate the **Average Treatment Effect**

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

Instead of trying to estimate

$$\mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill})]$$

let's try to estimate

$$\mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill}) | X]$$

---

## Properties (covariates) in our data

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

## Learning a policy

Based on said properties, we can estimate **Conditional Average Treatment Effects**:

1. Estimate $\hat{\tau}(X) = \mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill}) | X]$
2. Distribute pills according to this policy:
   $\pi(X) := \begin{cases} \text{blue pill} & \text{if } \hat{\tau}(X) \geq 0 \\ \text{red pill} & \text{if }\hat{\tau}(X) < 0 \end{cases}$

---

## But how?

Estimating $\hat{\tau}(X) = \mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill}) | X]$ is a really difficult statistical problem.

<!-- Why is it difficult? -->

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

- We can't know the Individual Treatment Effect (ITE).
- Yet, we can define an estimand, the Conditional Average Treatment Effect
  (CATE), which we can actually estimate:
  $$\tau(X) := \mathbb{E}[Y(\text{blue pill}) - Y(\text{red pill})|X]$$
- There are various of approaches for estimating CATEs.
- MetaLearners are a family of approaches for estimating CATEs.
  - Work by [Chernozhukov(2016)](https://arxiv.org/abs/1608.00060), [Nie(2017)](https://arxiv.org/pdf/1712.04912), [Kunzel(2017)](https://arxiv.org/pdf/1706.03461), [Kennedy(2020)](https://arxiv.org/pdf/2004.14497) and more

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
