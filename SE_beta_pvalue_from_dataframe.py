
# Rolling beta sensitivity and p-value analysis from an existing DataFrame


import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t as student_t
from statsmodels.tsa.stattools import coint


WINDOW = 120
PRED_HORIZON = 5
LARGE_MOVE_Q = 0.80
SE_BINS = 10
EPS = 1e-12

plt.rcParams.update({
    "figure.figsize": (11, 4.8),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "font.size": 11,
})


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def _prepare_pair(df, y_col, x_col):
    if df.index.name != "Date":
        raise ValueError(
            f"Expected df.index.name == 'Date', but got {df.index.name!r}. "
            "Set it with df.index.name = 'Date'."
        )

    missing = [c for c in [y_col, x_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    pair = df[[y_col, x_col]].copy()
    pair.index = pd.to_datetime(pair.index)
    pair = pair.sort_index()
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()

    if not pair.index.is_unique:
        pair = pair.groupby(level=0).last()

    return pair



def ols_window(xw, yw, eps=EPS):
    """OLS y = alpha + beta*x + u with the classical homoskedastic beta SE."""
    xw = np.asarray(xw, dtype=float)
    yw = np.asarray(yw, dtype=float)

    n = len(xw)
    if n <= 2:
        return (np.nan,) * 6

    xbar = xw.mean()
    ybar = yw.mean()

    xc = xw - xbar
    yc = yw - ybar

    sxx = np.sum(xc**2)
    if sxx <= eps:
        return (np.nan,) * 6

    beta = np.sum(xc * yc) / sxx
    alpha = ybar - beta * xbar

    resid = yw - alpha - beta * xw
    sigma2 = np.sum(resid**2) / (n - 2)

    beta_se = np.sqrt(sigma2 / sxx)

    if beta_se <= eps:
        beta_t = np.nan
        beta_p = np.nan
    else:
        beta_t = beta / beta_se
        beta_p = 2 * (1 - student_t.cdf(abs(beta_t), df=n - 2))

    return alpha, beta, beta_se, beta_t, beta_p, sigma2


def safe_coint_p(yw, xw):
    """Engle-Granger residual-based cointegration test."""
    try:
        stat, pval, _ = coint(yw, xw, trend="c", autolag="aic")
        return stat, pval
    except Exception:
        return np.nan, np.nan


def auc_manual(y_true, score):
    """Rank-based binary AUC without sklearn."""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    ranks = pd.Series(score).rank(method="average").to_numpy()

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    rank_sum_pos = ranks[y_true == 1].sum()

    return (
        rank_sum_pos - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)



def analyze_pair_sensitivity(
    df,
    y_col,
    x_col,
    window=WINDOW,
    pred_horizon=PRED_HORIZON,
    large_move_q=LARGE_MOVE_Q,
    se_bins=SE_BINS,
    out_dir=None,
    show_plots=True,
    save_plots=True,
):
    """
    Full rolling analysis for y_col on x_col.

    df must have a date index named 'Date'.
    """

    pair = _prepare_pair(df, y_col, x_col)

    if len(pair) < window + pred_horizon + 5:
        raise ValueError(
            f"Only {len(pair)} complete observations are available. "
            f"Need at least about {window + pred_horizon + 5}."
        )

    pair_label = f"{y_col} on {x_col}"
    pair_slug = f"{_safe_name(y_col)}_on_{_safe_name(x_col)}"

    if out_dir is None:
        out_dir = Path(f"presentation_plots_{pair_slug}")
    else:
        out_dir = Path(out_dir)

    if save_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    y = pair[y_col].to_numpy(dtype=float)
    x = pair[x_col].to_numpy(dtype=float)
    dates = pair.index

    rows = []

    for end in range(window - 1, len(pair)):
        start = end - window + 1

        xw = x[start:end + 1]
        yw = y[start:end + 1]

        alpha, beta, beta_se, beta_t, beta_p, sigma2 = ols_window(xw, yw)
        coint_stat, coint_p = safe_coint_p(yw, xw)

        sxx = np.sum((xw - np.mean(xw))**2)

        rows.append({
            "Date": dates[end],
            "alpha": alpha,
            "beta": beta,
            "beta_se": beta_se,
            "beta_t": beta_t,
            "beta_p": beta_p,
            "resid_var": sigma2,
            "sxx": sxx,
            "coint_stat": coint_stat,
            "coint_p": coint_p,
        })

    roll = pd.DataFrame(rows).set_index("Date")
    roll.index.name = "Date"

    roll["next_abs_beta_move"] = (
        roll["beta"].shift(-1) - roll["beta"]
    ).abs()

    future_moves = pd.concat(
        {
            h: (roll["beta"].shift(-h) - roll["beta"]).abs()
            for h in range(1, pred_horizon + 1)
        },
        axis=1,
    )

    roll["future_avg_beta_move"] = future_moves.mean(axis=1)
    roll["future_max_beta_move"] = future_moves.max(axis=1)

    q_move = roll["future_avg_beta_move"].quantile(large_move_q)
    roll["large_future_move"] = (
        roll["future_avg_beta_move"] >= q_move
    )

    eval_df = roll[
        [
            "beta_se",
            "beta_p",
            "beta_t",
            "future_avg_beta_move",
            "future_max_beta_move",
            "large_future_move",
        ]
    ].dropna().copy()

    eval_df["se_decile"] = (
        pd.qcut(
            eval_df["beta_se"],
            q=se_bins,
            labels=False,
            duplicates="drop",
        ) + 1
    )

    bin_summary = (
        eval_df
        .groupby("se_decile", observed=True)
        .agg(
            mean_se=("beta_se", "mean"),
            median_se=("beta_se", "median"),
            mean_future_move=("future_avg_beta_move", "mean"),
            median_future_move=("future_avg_beta_move", "median"),
            prob_large_move=("large_future_move", "mean"),
            n=("large_future_move", "size"),
        )
        .reset_index()
    )

    rho_spearman = (
        eval_df["beta_se"]
        .rank()
        .corr(eval_df["future_avg_beta_move"].rank())
    )
    rho_pearson = eval_df[
        "beta_se"
    ].corr(eval_df["future_avg_beta_move"])

    auc = pd.Series({
        "SE(beta)": auc_manual(
            eval_df["large_future_move"],
            eval_df["beta_se"],
        ),
        "beta p-value": auc_manual(
            eval_df["large_future_move"],
            eval_df["beta_p"],
        ),
        "-|t(beta)|": auc_manual(
            eval_df["large_future_move"],
            -eval_df["beta_t"].abs(),
        ),
    }, name="AUC")

    q80_se = eval_df["beta_se"].quantile(0.80)
    high_se = eval_df["beta_se"] >= q80_se

    result_summary = pd.DataFrame({
        "quantity": [
            "Spearman corr(SE, future beta movement)",
            "Pearson corr(SE, future beta movement)",
            "Mean future movement — lower 80% SE",
            "Mean future movement — top 20% SE",
            "P(large future movement) — lower 80% SE",
            "P(large future movement) — top 20% SE",
            "AUC(SE -> large future movement)",
        ],
        "value": [
            rho_spearman,
            rho_pearson,
            eval_df.loc[~high_se, "future_avg_beta_move"].mean(),
            eval_df.loc[high_se, "future_avg_beta_move"].mean(),
            eval_df.loc[~high_se, "large_future_move"].mean(),
            eval_df.loc[high_se, "large_future_move"].mean(),
            auc["SE(beta)"],
        ],
    })

    # 1) Rolling beta fluctuation
    fig, ax = plt.subplots(figsize=(11, 4.8))
    valid = roll[["beta", "beta_se"]].dropna()

    line = ax.plot(
        valid.index,
        valid["beta"],
        lw=1.5,
        label=rf"Rolling $\hat{{\beta}}_t$: {y_col} on {x_col}",
    )[0]

    ax.fill_between(
        valid.index,
        valid["beta"] - 1.96 * valid["beta_se"],
        valid["beta"] + 1.96 * valid["beta_se"],
        alpha=0.16,
        color=line.get_color(),
        label=r"$\hat{\beta}_t \pm 1.96\,SE_t$",
    )

    ax.set_title(
        f"Rolling coefficient fluctuation — {pair_label} "
        f"(window={window})"
    )
    ax.set_ylabel(r"$\hat{\beta}_t$")
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    if save_plots:
        fig.savefig(
            out_dir / f"01_beta_fluctuation_{pair_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 2) Rolling Engle-Granger p-value
    fig, ax = plt.subplots(figsize=(11, 4.8))
    valid = roll["coint_p"].dropna()

    ax.plot(
        valid.index,
        valid,
        lw=1.25,
        label=f"Engle-Granger p-value: {y_col} ~ {x_col}",
    )
    ax.axhline(
        0.05,
        ls="--",
        lw=1.25,
        label="5% threshold",
    )

    ax.set_ylim(0, 1)
    ax.set_title(
        f"Rolling cointegration p-value — {pair_label} "
        f"(window={window})"
    )
    ax.set_ylabel("Cointegration p-value")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_plots:
        fig.savefig(
            out_dir / f"02_cointegration_pvalue_{pair_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 3) Rolling coefficient p-value
    fig, ax = plt.subplots(figsize=(11, 4.8))
    valid = roll["beta_p"].dropna()

    ax.plot(
        valid.index,
        valid,
        lw=1.25,
        label=rf"$p$-value for $\beta=0$: {y_col} on {x_col}",
    )
    ax.axhline(
        0.05,
        ls="--",
        lw=1.25,
        label="5% threshold",
    )

    ax.set_ylim(0, 1)
    ax.set_title(
        f"Rolling coefficient-test p-value — {pair_label} "
        f"(window={window})"
    )
    ax.set_ylabel(r"$p$-value for $H_0:\beta=0$")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_plots:
        fig.savefig(
            out_dir / f"03_beta_pvalue_{pair_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 4) SE bin vs future beta movement
    fig, ax = plt.subplots(figsize=(8.6, 5.1))

    ax.plot(
        bin_summary["se_decile"],
        bin_summary["mean_future_move"],
        marker="o",
        lw=2,
    )

    ax.set_xticks(bin_summary["se_decile"])
    ax.set_xlabel("Current beta-SE quantile bin (low → high)")
    ax.set_ylabel(
        r"Mean future $|\hat{\beta}_{t+h}-\hat{\beta}_t|$"
    )
    ax.set_title(
        f"Current SE vs subsequent beta movement — {pair_label}\n"
        f"Spearman ρ={rho_spearman:.2f}; Pearson r={rho_pearson:.2f}"
    )
    fig.tight_layout()

    if save_plots:
        fig.savefig(
            out_dir / f"04_se_vs_future_beta_movement_{pair_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 5) Probability of large future movement
    fig, ax = plt.subplots(figsize=(8.6, 5.1))

    ax.bar(
        bin_summary["se_decile"],
        100 * bin_summary["prob_large_move"],
    )
    ax.axhline(
        100 * eval_df["large_future_move"].mean(),
        ls="--",
        lw=1.25,
        label=f"Unconditional top-{int((1-large_move_q)*100)}% rate",
    )

    ax.set_xticks(bin_summary["se_decile"])
    ax.set_xlabel("Current beta-SE quantile bin (low → high)")
    ax.set_ylabel("Probability of large future beta movement (%)")
    ax.set_title(
        f"Large future beta movements vs current SE — {pair_label}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_plots:
        fig.savefig(
            out_dir / f"05_large_move_probability_{pair_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 6) Predictor AUC
    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    ax.bar(auc.index, auc.values)
    ax.axhline(
        0.5,
        ls="--",
        lw=1.25,
        label="No discrimination",
    )

    finite_auc = auc[np.isfinite(auc)]
    upper = (
        max(0.85, float(finite_auc.max()) + 0.05)
        if len(finite_auc)
        else 0.85
    )
    ax.set_ylim(0.45, min(1.0, upper))
    ax.set_ylabel(
        f"AUC for top-{int((1-large_move_q)*100)}% future beta movement"
    )
    ax.set_title(
        f"Predictive ranking of coefficient-sensitivity measures — {pair_label}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_plots:
        fig.savefig(
            out_dir / f"06_predictor_auc_{pair_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    if save_plots:
        pair.to_csv(
            out_dir / f"pair_data_{pair_slug}.csv"
        )
        roll.to_csv(
            out_dir / f"rolling_results_{pair_slug}.csv"
        )
        bin_summary.to_csv(
            out_dir / f"se_bin_summary_{pair_slug}.csv",
            index=False,
        )
        result_summary.to_csv(
            out_dir / f"result_summary_{pair_slug}.csv",
            index=False,
        )
        auc.to_csv(
            out_dir / f"auc_{pair_slug}.csv",
            header=True,
        )

    print(f"Pair: {pair_label}")
    print(f"Complete observations: {len(pair):,}")
    print(f"Rolling window: {window}")
    print(f"Prediction horizon: {pred_horizon}")
    print(f"Large-move cutoff: {q_move:.6g}")
    print()
    print("Result summary:")
    display(result_summary)
    print()
    print("Predictor AUC:")
    display(auc.sort_values(ascending=False).to_frame())

    return {
        "pair": pair,
        "rolling": roll,
        "bin_summary": bin_summary,
        "auc": auc,
        "result_summary": result_summary,
        "out_dir": out_dir,
    }



# Run the TTF / German pair


Y_COL = "ttfy2027_2y"
X_COL = "ger_y2027_2y"

results = analyze_pair_sensitivity(
    df=df,
    y_col=Y_COL,
    x_col=X_COL,
    window=120,
    pred_horizon=5,
    large_move_q=0.80,
    se_bins=10,
    show_plots=True,
    save_plots=True,
)

roll = results["rolling"]
bin_summary = results["bin_summary"]
result_summary = results["result_summary"]
auc = results["auc"]



# Main outputs
