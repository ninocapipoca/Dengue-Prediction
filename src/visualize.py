"""Plotting helpers for the dengue exploratory notebook."""

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

# IBGE 2-digit UF (state) code -> name
UF_NAMES = {
    11: "Rondônia", 12: "Acre", 13: "Amazonas", 14: "Roraima", 15: "Pará",
    16: "Amapá", 17: "Tocantins", 21: "Maranhão", 22: "Piauí", 23: "Ceará",
    24: "Rio Grande do Norte", 25: "Paraíba", 26: "Pernambuco", 27: "Alagoas",
    28: "Sergipe", 29: "Bahia", 31: "Minas Gerais", 32: "Espírito Santo",
    33: "Rio de Janeiro", 35: "São Paulo", 41: "Paraná", 42: "Santa Catarina",
    43: "Rio Grande do Sul", 50: "Mato Grosso do Sul", 51: "Mato Grosso",
    52: "Goiás", 53: "Distrito Federal",
}

# 2-letter UF abbreviation -> name (SG_UF_NOT is stored this way pre-2007,
# and as the numeric IBGE code in UF_NAMES from 2007 on)
UF_ABBR = {
    "RO": "Rondônia", "AC": "Acre", "AM": "Amazonas", "RR": "Roraima", "PA": "Pará",
    "AP": "Amapá", "TO": "Tocantins", "MA": "Maranhão", "PI": "Piauí", "CE": "Ceará",
    "RN": "Rio Grande do Norte", "PB": "Paraíba", "PE": "Pernambuco", "AL": "Alagoas",
    "SE": "Sergipe", "BA": "Bahia", "MG": "Minas Gerais", "ES": "Espírito Santo",
    "RJ": "Rio de Janeiro", "SP": "São Paulo", "PR": "Paraná", "SC": "Santa Catarina",
    "RS": "Rio Grande do Sul", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso",
    "GO": "Goiás", "DF": "Distrito Federal",
}


def uf_label(value):
    """State name for a SG_UF_NOT value, whether it's an IBGE code (post-2007) or
    a 2-letter abbreviation (pre-2007)."""
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return UF_ABBR.get(value, value)
    return UF_NAMES.get(int(value), str(value))


# A distinct, stable colour per state (27 states -> tab20 + tab20b gives 40 colours).
# Keyed by state name so the same state gets the same colour whether SG_UF_NOT is an
# IBGE code (post-2007) or a 2-letter abbreviation (pre-2007).
_STATE_ORDER = sorted(set(UF_NAMES.values()))
_STATE_PALETTE = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
STATE_COLORS = {name: _STATE_PALETTE[i] for i, name in enumerate(_STATE_ORDER)}


def uf_color(value):
    """Stable, distinct colour for a SG_UF_NOT value (code or abbreviation)."""
    return STATE_COLORS.get(uf_label(value), "grey")


def plot_column_presence(wide, n_panels=4):
    """Heatmap of which columns are present per year, split across ``n_panels`` rows.

    ``wide`` is the frame from :func:`pipeline_funcs.column_presence` (a 'year'
    column plus one 0/1 column per column name). Increase ``n_panels`` to spread
    the many columns across more stacked panels for readability.
    """
    years = wide["year"].to_list()
    cols = wide.columns[1:]

    # ceiling division (-(-a // b)): columns per panel so we end up with ~n_panels panels
    chunk = -(-len(cols) // n_panels)
    chunks = [cols[i:i + chunk] for i in range(0, len(cols), chunk)]

    fig, axes = plt.subplots(
        len(chunks), 1,
        figsize=(chunk * 0.18, len(chunks) * len(years) * 0.3),
        squeeze=False,  # always 2D, so a single panel still iterates cleanly
    )

    for ax, sub in zip(axes[:, 0], chunks):
        mat = wide.select(sub).to_numpy()
        ax.pcolormesh(mat, cmap="Greys", edgecolors="lightgrey", linewidth=0.1)

        ax.set_xticks(np.arange(len(sub)) + 0.5, sub, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(years)) + 0.5, years, fontsize=8)
        ax.set_xlim(0, chunk)  # same width on every panel
        ax.invert_yaxis()

    fig.tight_layout()
    return fig, axes


def plot_state_timeseries(lf, year, pop_lf, per=100_000, state=None):
    """Monthly dengue notification rate (per ``per`` inhabitants) for one year.

    ``state`` selects which states to draw:
      - ``None``        -> every state
      - a single code   -> just that state, e.g. ``state=35``
      - a list of codes -> those states, e.g. ``state=[35, 33, 31]``
    Each state is drawn in its own distinct, stable colour (see ``STATE_COLORS``).
    """
    codes = None if state is None else ([state] if not isinstance(state, (list, tuple, set)) else list(state))
    if codes is not None:
        lf = lf.filter(pl.col("SG_UF_NOT").is_in(codes))

    pop_year = pop_lf.filter(pl.col("year") == int(year)).select("SG_UF_NOT", "population")

    data = (
        lf.with_columns(pl.col("DT_NOTIFIC").dt.truncate("1mo").alias("month"))
          .group_by("SG_UF_NOT", "month")
          .agg(pl.len().alias("cases"))
          .join(pop_year, on="SG_UF_NOT", how="left")
          .with_columns((pl.col("cases") / pl.col("population") * per).alias("rate"))
          .sort("SG_UF_NOT", "month")
          .collect()
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    for (code,), grp in data.group_by("SG_UF_NOT", maintain_order=True):
        ax.plot(grp["month"], grp["rate"], label=uf_label(code), color=uf_color(code))

    ax.set_xlabel("Month")
    ax.set_ylabel(f"Cases per {per:,} inhabitants")
    if codes is not None and len(codes) == 1:
        ax.set_title(f"Dengue notification rate in {uf_label(codes[0])}, {year}")
        ax.legend(fontsize=8, title="State")
    else:
        ax.set_title(f"Dengue notification rate by state, {year}")
        ax.legend(ncol=3, fontsize=8, title="State")
    fig.tight_layout()
    return fig, ax

def plot_residuals(train, fitted_model):
    """Four residual diagnostics for a fitted regression on the training rows: residuals vs fitted values,
    their distribution, residuals over time and residuals by calendar month. Returns (fig, axes)."""
    resid = fitted_model.resid
    fitted = fitted_model.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_fit, ax_hist, ax_time, ax_month = axes.flat

    ax_fit.scatter(fitted, resid, s=5, alpha=0.3)
    ax_hist.hist(resid, bins=50)
    ax_time.scatter(train["month"], resid, s=5, alpha=0.3)
    ax_month.boxplot([resid[train["month"].dt.month == m] for m in range(1, 13)])

    labels = [
        ("Fitted values, log-transformed", "Residual", "Residuals vs fitted"),
        ("Residual", "Number of state-months", "Distribution of residuals"),
        ("Date", "Residual", "Residuals over time"),
        ("Calendar month (1 = January)", "Residual", "Residuals by calendar month"),
    ]
    for ax, (xlabel, ylabel, title) in zip(axes.flat, labels):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    for ax in (ax_fit, ax_time, ax_month):
        ax.axhline(0, color="red", linewidth=1)

    return fig, axes
