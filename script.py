from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from git_root import git_root
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.patches import Patch
from metalearners import RLearner
from metalearners._utils import get_linear_dimension
from metalearners.data_generation import (
    compute_experiment_outputs,
    generate_treatment,
)
from metalearners.outcome_functions import linear_treatment_effect
from shap import TreeExplainer, summary_plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale

FONTSIZE = 22

RED = "#d40202"
BLUE = "#021bc2"


@cache
def results_dir() -> Path:
    return Path(git_root()) / "results"


def _covariates(rng, n_obs) -> tuple[pd.DataFrame, list[int]]:
    numerical_feature_names = ["age", "n_phil_books"]
    categorical_feature_names = ["personality_trait", "education", "atheist"]
    # We need to make sure that all numerical features are in the front.

    age = rng.poisson(lam=42, size=n_obs)
    n_phil_books = rng.poisson(lam=age / 10, size=n_obs)

    big_five = [
        "extraversion",
        "agreeableness",
        "openness",
        "conscientiousness",
        "neuroticism",
    ]
    personality_probabilities = [0.35, 0.3, 0.1, 0.2, 0.05]
    personality_trait = rng.choice(big_five, size=n_obs, p=personality_probabilities)

    education_levels = ["MS", "HS", "BSc", "BA", "MSc", "MA", "PhD"]
    education = np.empty(n_obs, dtype="U3")
    high_age_indices = age >= 25
    mid_age_indices = (age < 25) & (age >= 20)
    low_age_indices = age < 20
    education[high_age_indices] = rng.choice(
        education_levels, size=sum(high_age_indices)
    )
    education[mid_age_indices] = rng.choice(
        education_levels[:4], size=sum(mid_age_indices)
    )
    education[low_age_indices] = rng.choice(
        education_levels[:2], size=sum(low_age_indices)
    )

    atheism_base_line = 0.3
    # We want to model older people being less likely to be atheist.
    atheism_probabilities = atheism_base_line - minmax_scale(
        age, feature_range=(-0.5, 0.2)
    )
    atheist = rng.binomial(n=1, size=n_obs, p=atheism_probabilities)

    covariates = pd.DataFrame(
        {
            "age": age,
            "n_phil_books": n_phil_books,
            "personality_trait": personality_trait,
            "education": education,
            "atheist": atheist,
        }
    )
    for categorical_feature_name in categorical_feature_names:
        covariates[categorical_feature_name] = covariates[
            categorical_feature_name
        ].astype("category")
    categorical_indices = [
        len(numerical_feature_names) + i for i in range(len(categorical_feature_names))
    ]
    return covariates, categorical_indices


def data(rng, n_obs: int = 10_000, epsilon: float = 0.01):
    covariates, categorical_indices = _covariates(rng, n_obs)

    confounded_propensities = minmax_scale(
        covariates["age"],
        feature_range=(epsilon, 1 - epsilon),
    )

    treatments = generate_treatment(
        propensity_scores=confounded_propensities,
        rng=rng,
    )

    dim = get_linear_dimension(covariates)
    outcome_function = linear_treatment_effect(dim, ulow=-1, uhigh=2, rng=rng)
    potential_outcomes = outcome_function(covariates)

    observed_outcomes, true_cate = compute_experiment_outputs(
        mu=potential_outcomes,
        treatment=treatments,
        is_classification=False,
        rng=rng,
    )
    print("finished generating data")
    return (
        covariates,
        categorical_indices,
        treatments,
        confounded_propensities,
        observed_outcomes,
        true_cate,
    )


def hist_observed_outcomes(
    observed_outcomes: np.ndarray, treatments: np.ndarray
) -> None:
    fig, ax = plt.subplots()
    ax.hist(observed_outcomes)
    ax.set_xlabel("Observed outcomes y")
    fig.tight_layout()
    fig.savefig(results_dir() / "hist_observed_outcomes.png")

    fig, ax = plt.subplots()
    for w_value in np.unique(treatments):
        ax.hist(
            observed_outcomes[treatments == w_value], label=f"w={w_value}", alpha=0.5
        )
    ax.legend()
    ax.set_xlabel("Observed outcomes y by treatment assignment")
    fig.tight_layout()
    fig.savefig(results_dir() / "hist_observed_outcomes_by_treatment.png")


def md_correlation_matrix(
    numerical_covariates: pd.DataFrame, propensities: np.ndarray
) -> None:
    corr = numerical_covariates.corrwith(pd.Series(propensities))
    path = results_dir() / "corr_X_e.md"
    corr.to_markdown(path)


def plot_covariates_treatment(
    covariates: pd.DataFrame, categorical_indices: list[int], treatments: np.ndarray
) -> None:
    n_features = covariates.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_features)))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for index, feature in enumerate(covariates.columns):
        col_index = index % grid_size
        row_index = index // grid_size
        ax = axs[row_index, col_index]
        if index in categorical_indices:
            for treatment_value in np.unique(treatments):
                ax.hist(
                    covariates.loc[treatments == treatment_value, feature],
                    label=f"{treatment_value}",
                    alpha=0.5,
                )
            axs[row_index, col_index].set_xlabel(feature)
            ax.legend()

        else:
            violin_data = [
                covariates.loc[treatments == treatment_value, feature].values
                for treatment_value in np.unique(treatments)
            ]

            ax.violinplot(violin_data, positions=[0, 1], showmedians=True)
            axs[row_index, col_index].set_xlabel("treatment assignment")
            axs[row_index, col_index].set_ylabel(feature)
        ax.set_title(f"feature {feature}")

    fig.tight_layout()
    fig.savefig(results_dir() / "X_vs_w.png")


def ate_naive(treatments: np.ndarray, observed_outcomes: np.ndarray) -> float:
    return np.mean(observed_outcomes[treatments == 1]) - np.mean(
        observed_outcomes[treatments == 0]
    )


def ate_lin_reg(
    covariates: pd.DataFrame,
    categorical_indices: list[int],
    treatments: np.ndarray,
    observed_outcomes: np.ndarray,
) -> float:
    non_categorical_indices = [
        column_index
        for column_index in range(len(covariates.columns))
        if column_index not in categorical_indices
    ]

    covariates_ohe = pd.concat(
        (
            covariates.iloc[:, non_categorical_indices],
            pd.get_dummies(covariates.iloc[:, categorical_indices], drop_first=False),
            pd.Series(treatments, name="treatment"),
        ),
        axis=1,
    )

    # sklearn doesn't like it if some columns are ints and other strs.
    covariates_ohe.columns = [str(x) for x in covariates_ohe.columns]

    linear_regression = LinearRegression(fit_intercept=False).fit(
        X=covariates_ohe,
        y=observed_outcomes,
    )

    return linear_regression.coef_[-1]


def _ite_histogram(true_cate: np.ndarray, fontsize=FONTSIZE):
    fig, ax = plt.subplots(figsize=(10, 7))
    n, _, patches = ax.hist(true_cate, bins=20, color="grey")
    ax.set_xlabel("$\\tau$: difference in happiness \n (treatment effect)")

    for item in (
        [
            ax.title,
            ax.xaxis.label,
        ]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(FONTSIZE)
    return fig, ax, n, patches


def ite_histograms(true_cate: np.ndarray) -> None:
    fig, ax, n, patches = _ite_histogram(true_cate)
    fig.tight_layout()
    fig.savefig(results_dir() / "ites_plain.png")

    fig, ax, n, patches = _ite_histogram(true_cate)

    for i in range(len(n)):
        x_left = patches[i].get_x()
        x_right = x_left + patches[i].get_width()
        if x_right < 0:
            patches[i].set_facecolor(RED)
        elif x_left >= 0:
            patches[i].set_facecolor(BLUE)
        else:
            patches[i].set_facecolor("grey")

    ax.axvline(0, color="orange")
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor=BLUE, label="blue pillers"),
            Patch(facecolor=RED, edgecolor=RED, label="red pillers"),
            Patch(facecolor="grey", edgecolor="grey", label="on the edge"),
        ],
        prop={"size": FONTSIZE},
    )

    fig.tight_layout()
    fig.savefig(results_dir() / "ites_classified.png")


def fit_metalearner(covariates, treatments, observed_outcomes):
    rlearner = RLearner(
        nuisance_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=LGBMRegressor,
        nuisance_model_params={"verbose": -1},
        treatment_model_params={"verbose": -1},
        propensity_model_params={"verbose": -1},
        is_classification=False,
        n_variants=2,
    )
    rlearner.fit(
        X=covariates,
        y=observed_outcomes,
        w=treatments,
    )

    return rlearner


def shap_values(learner, covariates):
    plt.clf()
    explainer = learner.explainer()
    shap_values = explainer.shap_values(covariates, TreeExplainer)
    summary_plot(shap_values[0], features=covariates, show=False)
    plt.savefig(results_dir() / "shap_values.png")


if __name__ == "__main__":
    rng = np.random.default_rng(1337)
    (
        covariates,
        categorical_indices,
        treatments,
        propensities,
        observed_outcomes,
        true_cate,
    ) = data(rng)

    hist_observed_outcomes(observed_outcomes, treatments)

    numerical_covariates = covariates.iloc[
        :, [c for c in range(len(covariates.columns)) if c not in categorical_indices]
    ]

    plot_covariates_treatment(
        covariates,
        categorical_indices,
        treatments,
    )
    md_correlation_matrix(
        numerical_covariates,
        propensities,
    )

    ate_estimate_naive = ate_naive(
        treatments=treatments, observed_outcomes=observed_outcomes
    )
    print(f"{ate_estimate_naive=}")

    ate_estimate_lin_reg = ate_lin_reg(
        covariates=covariates,
        categorical_indices=categorical_indices,
        treatments=treatments,
        observed_outcomes=observed_outcomes,
    )
    print(f"{ate_estimate_lin_reg=}")

    ite_histograms(true_cate)

    rlearner = fit_metalearner(covariates, treatments, observed_outcomes)

    shap_values(rlearner, covariates)
