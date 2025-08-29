#%%
import data_loader as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import dowhy
from dowhy import CausalModel
import pandas as pd
from sklearn.model_selection import train_test_split
from tableone import TableOne
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
#%%
random_seed= 42
#%%

df= pd.read_csv('data/cleand_df2018.csv')
df.head()
#%%
#save df as csv
# df.to_csv('cleand_df2018.csv', index=False)
#%%
treatment = "computer"
outcome = "achievement"

country_cols = [col for col in df.columns if col.startswith("country_")]




confounders = [
    'escs', 'mother_educ', 'father_educ', 'desk', 'room', 'book', 'gender'
] + country_cols


#%%
X=df[confounders]
y=df[outcome]
T=df[treatment]

#%%
X_train, X_val, T_train, T_val, y_train, y_val = train_test_split(X, T, y, test_size=0.3, random_state=random_seed)
#%%

def train_propensity_model(model, X_train, T_train):
    """
    Return a fitted propensity model on your training data
    :param model:
    :param X_train:
    :param T_train:
    :return:
    """
    return model.fit(X_train, T_train)

def eval_propensity_model(fitted_model, X_val, T_val,model_name="Propensity Model"):
    """

    :param fitted_model:
    :param X_val:
    :param T_val:
    :return:
    """


    # Get the predicted probabilities
    y_pred = fitted_model.predict_proba(X_val)[:, 1]

    # Calculate the Brier score
    brier_score = brier_score_loss(T_val, y_pred)
    print('' + model_name + ' evaluation:')
    # Print the Brier score
    print(f'Brier score: {brier_score}')

    # Plot the calibration curve
    prob_true, prob_pred = calibration_curve(T_val, y_pred, n_bins=10)

    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.legend()
    plt.show()

#%%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    log_loss, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import pandas as pd
import joblib
import matplotlib.pyplot as plt

models = {
    "Logistic Regression CV": LogisticRegressionCV(cv=5, max_iter=1000, random_state=random_seed),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_seed),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_seed),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=random_seed),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_seed),
    "CatBoost": CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0, random_state=random_seed),

}

results = {}

for name, model in models.items():
    model.fit(X_train, T_train)
    probas = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    auc = roc_auc_score(T_val, probas)
    brier = brier_score_loss(T_val, probas)
    acc = accuracy_score(T_val, preds)
    ll = log_loss(T_val, probas)
    f1 = f1_score(T_val, preds)
    precision = precision_score(T_val, preds)
    recall = recall_score(T_val, preds)
    results[name] = {
        'Brier': brier
        }
    print(f"{name}:  Brier={brier:.4f}")

df_results = pd.DataFrame(results).T
print("\nAll results:\n", df_results)


best_brier_model = df_results['Brier'].idxmin()



print(f"Best model by Brier: {best_brier_model} ({df_results.loc[best_brier_model, 'Brier']:.4f})")

# Save the best model
best_model = models[best_brier_model]
# Calibration curve for the best model
probas = best_model.predict_proba(X_val)[:, 1]

prob_true, prob_pred = calibration_curve(T_val, probas, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title(f'Calibration curve ({best_brier_model})')
plt.legend()
plt.show()

#%%
def check_overlap(propensity_model, X_train, T_train,):
    """
    Show histograms of the propensity on the T=0 and the T=1 group

    :param propensity_model:
    :param X_train:
    :param T_train:
    :return:
    """
    # Get the predicted probabilities
    y_pred = propensity_model.predict_proba(X_train)[:, 1]

    # Create histograms for T=0 and T=1
    plt.figure(figsize=(10, 6))

    sns.histplot(y_pred[T_train == 1], color='red', label='Treated (Has Computer)', bins=30, stat="density", alpha=0.6)
    sns.histplot(y_pred[T_train == 0], color='blue', label='Control (No Computer)', bins=30, stat="density", alpha=0.6)

    plt.xlabel('Predicted Probability of Treatment')
    plt.ylabel('Density')
    plt.title('Overlap Check: Predicted Probabilities for T=0 and T=1')


    plt.legend()
    plt.show()

#%%
check_overlap(best_model, X_train, T_train)
#%%
def trim_by_propensity(X, y, T, propensity_scores, lower=0.05, upper=0.95):
    mask = (propensity_scores >= lower) & (propensity_scores <= upper)
    X_trim = X[mask]
    y_trim = y[mask]
    T_trim = T[mask]
    propensity_scores_trim = propensity_scores[mask]
    print(f"Trimmed to {mask.sum()} samples (from {len(mask)}) in [{lower}, {upper}] region.")
    return X_trim, y_trim, T_trim, propensity_scores_trim

propensity_scores = best_model.predict_proba(X)[:, 1]

# Trim the data
X_trim, y_trim, T_trim, propensity_scores_trim = trim_by_propensity(
    X, y, T, propensity_scores, lower=0.05, upper=0.95
)
#%%
X_train, X_val, T_train, T_val, y_train, y_val = train_test_split(X_trim, T_trim, y_trim, test_size=0.3, random_state=random_seed)
#%%
#re train the best model on the trimmed data
best_model = train_propensity_model(best_model, X_train, T_train)
#brier score on the trimmed data
eval_propensity_model(best_model, X_val, T_val, model_name="Best Propensity Model (Trimmed)")
#%% md
# T-learner
#%%
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_squared_error
)
from sklearn.base import clone
import joblib

def train_t_learner(model, X_train, y_train):
    return model.fit(X_train, y_train)

from tqdm.auto import tqdm

def t_learner_model_selection(models, X_train, y_train, X_val, y_val):
    metrics = {
        "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
    }
    results = {name: {} for name in [m.__class__.__name__ for m in models]}
    best_models = {}

    for metric_name, metric_func in tqdm(metrics.items(), desc="Metrics"):
        best_score = np.inf if metric_name in ["MSE"] else -np.inf
        best_model = None

        for model in tqdm(models, desc=f"Models for {metric_name}", leave=False):
            fitted_model = train_t_learner(clone(model), X_train, y_train)
            y_pred = fitted_model.predict(X_val)
            score = metric_func(y_val, y_pred)
            results[model.__class__.__name__][metric_name] = score

            is_better = (score < best_score) if metric_name in ["MSE"] else (score > best_score)
            if is_better:
                best_score = score
                best_model = fitted_model

        best_models[metric_name] = best_model
        print(f'Best model for {metric_name}: {best_model.__class__.__name__} with score {best_score:.4f}')

    # Display results table
    df_results = pd.DataFrame(results).T
    print("All validation scores:")
    print(df_results)

    for metric_name in metrics:
        print(f"Best model by {metric_name}: {best_models[metric_name].__class__.__name__} (Score: {df_results[metric_name][best_models[metric_name].__class__.__name__]:.4f})")

    # Save best model by MSE
    joblib.dump(best_models["MSE"], "best_t_learner_model_MSE.pkl")
    print(f"\nSaved best T-learner model (by MSE): {best_models['MSE'].__class__.__name__}")

    return best_models


models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0),
    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0),
    XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0),
    LGBMRegressor(n_estimators=100, max_depth=3, random_state=0),

]

# For treated
X_train_treated = X_train[T_train == 1]
y_train_treated = y_train[T_train == 1]
X_val_treated = X_val[T_val == 1]
y_val_treated = y_val[T_val == 1]
# For control
X_train_control = X_train[T_train == 0]
y_train_control = y_train[T_train == 0]
X_val_control = X_val[T_val == 0]
y_val_control = y_val[T_val == 0]
print("\n=== Selecting model for T=1  ===")
best_models_treated = t_learner_model_selection(models, X_train_treated, y_train_treated, X_val_treated, y_val_treated)

print("\n=== Selecting model for T=0  ===")
best_models_control = t_learner_model_selection(models, X_train_control, y_train_control, X_val_control, y_val_control)

#%%

def calculate_t_learner_ate(model_treated, model_control, X):
    """
    Estimate the Average Treatment Effect (ATE) using a fitted T-learner.

    Args:
        model_treated: Fitted model for treated (T=1) group.
        model_control: Fitted model for control (T=0) group.
        X: DataFrame of covariates to estimate potential outcomes on.

    Returns:
        ate (float): Estimated ATE on X.
    """
    y1_pred = model_treated.predict(X)
    y0_pred = model_control.predict(X)
    ate = (y1_pred - y0_pred).mean()
    print(f"Estimated ATE (T-Learner): {ate:.4f}")
    return ate

def t_learner_confidence_intervals(model_treated, model_control,
                                   X_train, T_train, y_train,
                                   X_pred, n_iterations=1000, alpha=0.05, random_seed=random_seed):
    """
    Estimate a 95% bootstrap confidence interval for the T-learner ATE.

    Args:
        model_treated: Fitted model for T=1.
        model_control: Fitted model for T=0.
        X_train: Training features (DataFrame)
        T_train: Treatment vector (Series or array)
        y_train: Outcome vector (Series or array)
        X_pred: Covariates to predict the ATE on (e.g., validation or full set)
        n_iterations: Number of bootstrap samples.
        alpha: Significance level (default=0.05 for 95% CI)
        random_seed: Reproducibility.

    Returns:
        (lower, upper): Lower and upper bounds of the bootstrap CI.
    """
    np.random.seed(random_seed)
    ate_estimates = []

    for _ in tqdm(range(n_iterations), desc=f"iterations"):
        # Bootstrap sample indices
        indices = np.random.choice(X_train.index, size=len(X_train), replace=True)
        X_resampled = X_train.loc[indices]
        T_resampled = T_train.loc[indices]
        y_resampled = y_train.loc[indices]

        # Split resampled data by treatment group
        X1, y1 = X_resampled[T_resampled == 1], y_resampled[T_resampled == 1]
        X0, y0 = X_resampled[T_resampled == 0], y_resampled[T_resampled == 0]

        # Refit both models on bootstrap sample
        model1 = clone(model_treated).fit(X1, y1)
        model0 = clone(model_control).fit(X0, y0)

        # Predict on X_pred
        y1_pred = model1.predict(X_pred)
        y0_pred = model0.predict(X_pred)
        ate_estimates.append((y1_pred - y0_pred).mean())

    # Calculate the confidence interval
    lower = np.percentile(ate_estimates, 100 * alpha / 2)
    upper = np.percentile(ate_estimates, 100 * (1 - alpha / 2))

    print(f"95% Confidence Interval for ATE (T-Learner): [{lower:.4f}, {upper:.4f}]")
    return lower, upper

#%%
print(best_models_treated)
#%%
ate_t_learner = calculate_t_learner_ate(best_models_treated['MSE'], best_models_control['MSE'], X_val)

# Get confidence interval:
t_learner_ci= t_learner_confidence_intervals(
    best_models_treated['MSE'], best_models_control['MSE'],
    X_train, T_train, y_train, X_val, n_iterations=1000, alpha=0.05, random_seed=random_seed
)

#%% md
# IPW
#%%


def calculate_ipw_ate(propensity_scores, T, y):
    """
    Estimate ATE using Inverse Probability Weighting (IPW).

    Args:
        propensity_scores: np.array or pd.Series of predicted propensities (P(T=1|X))
        T: Treatment indicator (np.array or pd.Series, 0/1)
        y: Outcome (np.array or pd.Series)

    Returns:
        ipw_ate (float): Estimated ATE
    """

    # ps = np.clip(propensity_scores, epsilon, 1 - epsilon)
    weight_treated = T / propensity_scores
    weight_control = (1 - T) / (1 - propensity_scores)

    ipw_ate = np.mean(weight_treated * y) - np.mean(weight_control * y)
    print(f"Estimated ATE (IPW): {ipw_ate:.4f}")
    return ipw_ate

#%%

propensity_scores_val = best_model.predict_proba(X_val)[:, 1]
ipw_ate = calculate_ipw_ate(propensity_scores_val, T_val, y_val)


#%%
from tqdm.auto import tqdm
import numpy as np

def ipw_confidence_intervals(model, X_val, T_val, y_val, n_iterations=1000, alpha=0.05, random_seed=random_seed, verbose=True):
    """
    Bootstrap 95% confidence interval for IPW ATE.

    Args:
        model: fitted propensity model (must have predict_proba)
        X_val: validation features (DataFrame)
        T_val: validation treatment (Series)
        y_val: validation outcome (Series)
        n_iterations: number of bootstrap iterations (default 1000)
        alpha: significance level (default 0.05 for 95% CI)
        random_seed: random seed for reproducibility
        verbose: whether to print the CI

    Returns:
        lower, upper: confidence interval for IPW ATE
    """
    np.random.seed(random_seed)
    ate_estimates = []
    epsilon = 1e-6

    for _ in tqdm(range(1000), desc=f"IPW Bootstrap"):
        indices = np.random.choice(X_val.index, size=len(X_val), replace=True)
        X_boot = X_val.loc[indices]
        T_boot = T_val.loc[indices]
        y_boot = y_val.loc[indices]

        # Predict propensity scores
        p_scores =model.predict_proba(X_boot)[:, 1]

        # Compute IPW ATE for this sample
        weights_treated = T_boot / p_scores
        weights_control = (1 - T_boot) / (1 - p_scores)
        ipw_ate = np.mean(weights_treated * y_boot) - np.mean(weights_control * y_boot)
        ate_estimates.append(ipw_ate)

    lower = np.percentile(ate_estimates, 100 * alpha / 2)
    upper = np.percentile(ate_estimates, 100 * (1 - alpha / 2))

    if verbose:
        print(f"95% Confidence Interval for ATE (IPW): [{lower:.4f}, {upper:.4f}]")
    return lower, upper

#%%
ipw_ci = ipw_confidence_intervals(best_model, X_val, T_val, y_val)

#%% md
# doubly robust
#%%


def calculate_dr_ate(y, T, propensity_scores, mu0_pred, mu1_pred):
    """
    Doubly Robust ATE Estimation.

    Args:
        y:      Observed outcome
        T:      Treatment indicator (
        propensity_scores:  estimated propensity
        mu0_pred:  Predicted outcome  untreated
        mu1_pred:  Predicted outcome  treated

    Returns:
        dr_ate:  Estimated ATE
    """

    # eps = 1e-6
    # propensity_scores = np.clip(propensity_scores, eps, 1-eps)

    # Doubly Robust estimate:
    dr_scores = (mu1_pred - mu0_pred) \
        + T * (y - mu1_pred) / propensity_scores \
        - (1 - T) * (y - mu0_pred) / (1 - propensity_scores)

    dr_ate = np.mean(dr_scores)
    print(f"Estimated ATE (Doubly Robust): {dr_ate:.4f}")
    return dr_ate

mu1_pred = best_models_treated['MSE'].predict(X_trim)
mu0_pred = best_models_control['MSE'].predict(X_trim)

dr_ate = calculate_dr_ate(y_trim, T_trim, propensity_scores_trim, mu0_pred, mu1_pred)

#%%

from sklearn.base import clone
from tqdm.auto import tqdm

def dr_confidence_intervals(
    X, y, T,
    propensity_model, model_treated, model_control,
    n_iterations=1000, alpha=0.05, random_seed=random_seed, verbose=True
):
    """
    Bootstrap 95% confidence interval for Doubly Robust ATE.

    Args:
        X: Features  already trimmed
        y: Outcome
        T: Treatment
        propensity_model: fitted propensity score model
        model_treated: fitted outcome model for T=1
        model_control: fitted outcome model for T=0
        n_iterations: number of bootstrap samples
        alpha: significance level
        random_seed: for reproducibility
        verbose: whether to print the CI

    Returns:
        (lower, upper): bounds of the confidence interval
    """
    np.random.seed(random_seed)
    ate_estimates = []
    eps = 1e-6

    for _ in tqdm(range(n_iterations), desc="DR Bootstrap"):
        # Sample with replacement
        indices = np.random.choice(X.index, size=len(X), replace=True)
        X_boot = X.loc[indices]
        y_boot = y.loc[indices]
        T_boot = T.loc[indices]

        # Estimate propensity
        # p_scores = np.clip(propensity_model.predict_proba(X_boot)[:, 1], eps, 1-eps)
        p_scores=propensity_model.predict_proba(X_boot)[:, 1]
        # Predict outcomes as if all treated / control
        mu1_pred = model_treated.predict(X_boot)
        mu0_pred = model_control.predict(X_boot)

        # DR estimate
        dr_scores = (mu1_pred - mu0_pred) \
            + T_boot * (y_boot - mu1_pred) / p_scores \
            - (1 - T_boot) * (y_boot - mu0_pred) / (1 - p_scores)
        dr_ate = np.mean(dr_scores)
        ate_estimates.append(dr_ate)

    lower = np.percentile(ate_estimates, 100 * alpha / 2)
    upper = np.percentile(ate_estimates, 100 * (1 - alpha / 2))
    if verbose:
        print(f"95% Confidence Interval for ATE (Doubly Robust): [{lower:.4f}, {upper:.4f}]")
    return lower, upper

dr_ci = dr_confidence_intervals(
    X_trim, y_trim, T_trim,
    propensity_model=best_model,
    model_treated=best_models_treated['MSE'],
    model_control=best_models_control['MSE'],
    alpha=0.05,
    random_seed=random_seed
)

#%% md
# matching
#%%
import pandas as pd
from tableone import TableOne

T_train_named = T_train.copy()
T_train_named.name = 'T'

# Concatenate along columns
df_table = pd.concat([X_train, T_train_named], axis=1)
#  the columns
columns = confounders

table1 = TableOne(df_table, columns=columns, groupby='T', nonnormal=[], pval=False, smd=True)

# Print the summary table
print(table1.tabulate(tablefmt="fancy_grid"))


#%%
def compute_smd(col, treated, control):
    """Compute standardized mean difference for a covariate."""
    mean_t, mean_c = treated[col].mean(), control[col].mean()
    std_t, std_c = treated[col].std(), control[col].std()
    pooled_std = np.sqrt((std_t ** 2 + std_c ** 2) / 2)
    return abs(mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

def plot_smd_balance(data, covariates, treatment_col='T', threshold=0.1):
    """Plot SMD for covariates with a horizontal layout like in the tutorial."""
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]

    smd_scores = {
        cov: compute_smd(cov, treated, control) for cov in covariates
    }

    # Sort covariates for better display
    covs, scores = zip(*sorted(smd_scores.items(), key=lambda x: x[1], reverse=True))

    # Plot horizontal bar chart
    plt.figure(figsize=(8, 15))
    plt.barh(covs, scores, color='skyblue')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title("SMD for Covariates After Matching")
    plt.xlabel("Standardized Mean Difference (SMD)")
    plt.ylabel("Covariates")
    plt.legend()
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

plot_smd_balance(data=pd.concat([X_train, T_train], axis=1), covariates= confounders,treatment_col=treatment)
#%%
import numpy as np
import pandas as pd

# X_trim, y_trim, T_trim, propensity_scores_trim
df = pd.DataFrame(X_trim.copy())
df['y'] = y_trim
df['T'] = T_trim
df['ps'] = propensity_scores_trim
df = df.reset_index(drop=True)

#%%
from sklearn.neighbors import NearestNeighbors

caliper = 0.05

treated = df[df['T'] == 1].copy()
control = df[df['T'] == 0].copy()

# For each treated find a control within caliper
matched_treated_idx = []
matched_control_idx = []

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[['ps']])

for i, row in treated.iterrows():
    dist, idx = nbrs.kneighbors([[row['ps']]])
    if dist[0][0] <= caliper:
        matched_treated_idx.append(i)
        matched_control_idx.append(control.index[idx[0][0]])

# Build matched dataset
matched_df = pd.concat([
    treated.loc[matched_treated_idx],
    control.loc[matched_control_idx]
], axis=0).sort_index()

print(f"Matched {len(matched_treated_idx)} treated-control pairs using caliper {caliper}")
#%%
# # Check SMD before/after matching (optional, using TableOne for example)
# from tableone import TableOne
# matched_df = matched_df.reset_index(drop=True)
# print(TableOne(matched_df, columns=X_trim.columns.tolist(), groupby='T', smd=True).tabulate(tablefmt="fancy_grid"))


#%%
# Split matched data back to treated/control
matched_treated = matched_df[matched_df['T'] == 1]
matched_control = matched_df[matched_df['T'] == 0]

ate_matched = (matched_treated['y'].values - matched_control['y'].values).mean()
print(f"ATE from caliper matched sample: {ate_matched:.4f}")

#%%
import numpy as np
from tqdm.auto import tqdm

def caliper_matched_ci(matched_treated, matched_control, n_iterations=1000, alpha=0.05, random_seed=random_seed, verbose=True):
    """
    Bootstrap confidence interval for the ATE from caliper matching.

    Args:
        matched_treated: DataFrame of matched treated units, aligned to matched_control
        matched_control: DataFrame of matched control units, aligned
        n_iterations: Number of bootstrap samples
        alpha: Significance level
        random_seed: For reproducibility
        verbose: Print the CI

    Returns:
        (lower, upper): bounds of the confidence interval
    """
    np.random.seed(random_seed)
    n = len(matched_treated)
    ate_estimates = []
    for _ in tqdm(range(n_iterations), desc="Caliper matching bootstrap"):
        idx = np.random.choice(n, n, replace=True)
        ate = (matched_treated['y'].values[idx] - matched_control['y'].values[idx]).mean()
        ate_estimates.append(ate)
    lower = np.percentile(ate_estimates, 100 * alpha / 2)
    upper = np.percentile(ate_estimates, 100 * (1 - alpha / 2))
    if verbose:
        print(f"95% Confidence Interval for ATE (Caliper matching): [{lower:.4f}, {upper:.4f}]")
    return lower, upper

#%%
matching_ci = caliper_matched_ci(matched_treated, matched_control, n_iterations=1000, alpha=0.05, random_seed=random_seed)

#%%
# X for matched treated and controls
matched_X_treated = treated.loc[matched_treated_idx, X_trim.columns]
matched_X_control = control.loc[matched_control_idx, X_trim.columns]

# Stack them together for all matched samples
matched_X = pd.concat([matched_X_treated, matched_X_control], axis=0)
matched_X = matched_X.reset_index(drop=True)

# Build T for all matched samples
matched_T = np.concatenate([np.ones(len(matched_X_treated)), np.zeros(len(matched_X_control))])
matched_T = pd.Series(matched_T, name='T')

print(matched_X.shape, matched_T.shape)

#%%
plot_smd_balance(data=pd.concat([matched_X, matched_T], axis=1), covariates= confounders)
#%%
check_overlap(best_model,matched_X,matched_T)
#%%

#%%
methods = ["DR", "IPW", "Matching",'T_learner']
cis = [dr_ci, ipw_ci, matching_ci,t_learner_ci]

# Midpoints and error bars
midpoints = [(low + high) / 2 for low, high in cis]
errors = [(high - low) / 2 for low, high in cis]

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(midpoints, methods, xerr=errors, fmt='o', capsize=5, color='teal', ecolor='lightcoral')
plt.axvline(0, color='gray', linestyle='--', label="Zero Effect")

plt.xlabel("Estimated Treatment Effect ")
plt.title("95% Confidence Intervals for Treatment Effect")
plt.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
#%%
data=[('Matching-cliper',ate_matched,matching_ci[0],matching_ci[1]),
      ('Doubly Robust',dr_ate,dr_ci[0],dr_ci[1]),
('IPW',ipw_ate,ipw_ci[0],ipw_ci[1]),
('T-learner',ate_t_learner,t_learner_ci[0],t_learner_ci[1])]




df = pd.DataFrame(data, columns=["label", "ate", "lo", "hi"])

# plot
fig, ax = plt.subplots(figsize=(7, 5), dpi=130)

x = np.arange(len(df))

# error bars (CIs)
ax.errorbar(
    x,
    df["ate"],
    yerr=[df["ate"] - df["lo"], df["hi"] - df["ate"]],
    fmt='o',                   # circular marker at the ATE
    ms=4,                      # marker size
    lw=2,                      # line (error bar) width
    elinewidth=2,              # error bar line width
    capsize=4,                 # small caps at error bar ends
    color="0.3",               # dark gray marker/line
    ecolor="0.6",              # lighter gray for error bars
)

# value labels in boxes at each point
for xi, y in zip(x, df["ate"]):
    ax.text(
        xi, y,
        f"{y:.2f}",
        va="center", ha="center",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=0.8)
    )

# x-axis labels on two lines
ax.set_xticks(x)
ax.set_xticklabels(df["label"], fontsize=9)

# y-axis label and title
ax.set_ylabel("Estimated Treatment Effect")
ax.set_title("95% Confidence Intervals for Treatment Effect", pad=8)

# y-limits with a bit of padding
ymin = min(df["lo"].min(), (df["ate"].min())) - 1.0
ymax = max(df["hi"].max(), (df["ate"].max())) + 1.0
ax.set_ylim(ymin, ymax)

# light grid
ax.grid(axis="y", ls="-", lw=0.5, color="0.9")
plt.axhline(
    y=0,
    color='gray',
    linestyle='--',
    linewidth=1.2,
    label="Zero Effect"
)

# tighten layout
plt.tight_layout()
plt.show()
#%%
