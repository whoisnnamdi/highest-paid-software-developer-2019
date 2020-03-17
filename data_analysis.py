import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import data_prep

def unlog(logged):
    """
    Unlogs logged quantities
    
    """
    return np.e**(logged) - 1

def double_selection(df: pd.DataFrame, 
                     controls: dict, 
                     controls_adj: list, 
                     category: str) -> pd.DataFrame:
    """
    Runs double selection on dataframe for provided category, selecting relevant covariates
    
    Returns dataframe with important covariates along with original category
    """

    X = df[controls_adj].copy()

    W = df.drop(controls_adj, axis=1).assign(const=1).copy()

    W = W.drop("Income", axis=1)

    Y = df["Income"].copy()

    clf = LassoCV(cv=5, max_iter=10000, selection="random", n_jobs=-1)

    sfm = SelectFromModel(clf)

    for i, X_j in enumerate(X.columns):
        if i==0:
            A = sfm.fit(W, X[X_j]).get_support()
        else:
            A = A | sfm.fit(W, X[X_j]).get_support()

    B = sfm.fit(W, Y).get_support()
    
    """
    Code for square root lasso
    
    from scipy.stats import norm
    
    n = len(Y)
    p = W.shape[1]
    
    print(n, p)
    
    alpha = 1.1 * np.sqrt(n) * norm.ppf(1 - 0.05 / (2 * p))
    
    return sm.OLS(X["DevType_back_end"].astype(float), W.astype(float)).fit_regularized(method="sqrt_lasso", alpha=alpha).params
    
    """

    return pd.concat([W.T[A | B].T, X], axis=1)

def explain(df: pd.DataFrame, 
            regression, 
            controls: dict, 
            cat_to_explain: str, 
            coef_to_explain: str, 
            top: int = 5) -> (pd.DataFrame, float):
    """
    Decomposes explained portion of income gap between groups and returns top explainers

    Decomposition follows methodology of (Gelbach 2016)

    Returns: Dataframe with top 3 explainers and respective explained gaps
    """

    name = cat_to_explain + "_" + coef_to_explain

    df_copy = df.copy()

    X_1 = data_prep.prune_df(df_copy, cat_to_explain).assign(const=1)

    explainers = [control for control in controls.keys() if control != cat_to_explain]

    results = pd.DataFrame()
    
    for i, explainer in enumerate(explainers):

        X_2 = data_prep.prune_df(df_copy, explainer)

        X_2 = X_2.reindex(sorted(X_2.columns), axis=1)

        B_2 = regression.params.filter(like=explainer+"_").sort_index()

        X_2 = X_2[B_2.index]

        H_k = X_2.values @ B_2

        if i == 0:
            results = pd.DataFrame(sm.OLS(endog=H_k, exog=X_1).fit().params.rename(explainer))
        else:
            d = sm.OLS(endog=H_k, exog=X_1).fit().params.rename(explainer)

            results = results.join(d)

    neg = results.T.filter(items=[name]).sort_values(by=name)[name].sum() < 0

    results = results.T.filter(items=[name]).sort_values(by=name, ascending=neg)

    results.columns = [coef_to_explain]

    return results.iloc[:top].append(results.iloc[top:].sum().rename("Other")), regression.params.filter(items=[name]).values[0]