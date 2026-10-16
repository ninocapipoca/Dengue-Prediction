"""Data loading and cleaning helpers"""

import json
import os
import urllib.request
from datetime import date

import polars as pl
import polars.selectors as cs

import pathlib as Path
import numpy as np

# ------------ Project folders ------------------------------------------------------
# src/ holds the code, data/ the downloaded raw data and output/ everything the code writes
PROJECT_DIR = Path.Path(os.path.relpath(Path.Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SINAN_DIR = DATA_DIR / "SINAN_data"
UNIPLU_DIR = DATA_DIR / "UNIPLU_data"
MUNICIPALITY_CSV = DATA_DIR / "Tabela_de_Municipios_corrigida.csv"
RAINFALL_CLEANED_DIR = OUTPUT_DIR / "Rainfall_cleaned"
POPULATION_CSV = OUTPUT_DIR / "pop_estimates.csv"

SINAN_URL = "https://www.kaggle.com/datasets/gallo33henrique/dengue-no-brasil-sinan-20002025"
UNIPLU_URL = "https://doi.org/10.5281/zenodo.18883358"
MUNICIPALITY_URL = "https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/orgaos-e-municipios"

# ------------ Variables reused across notebooks ------------------------------------
# Codes reused across the pipeline
DENGUE_CONFIRMADO = [1, 10, 11, 12]  # confirmed-dengue classification codes

# Pre-2007 files store SG_UF_NOT as a 2-letter abbreviation; from 2007 on it is the
# numeric IBGE code. This maps the abbreviations to those codes so SG_UF_NOT is uniform.
UF_ABBR_TO_CODE = {
    "RO": 11, "AC": 12, "AM": 13, "RR": 14, "PA": 15, "AP": 16, "TO": 17,
    "MA": 21, "PI": 22, "CE": 23, "RN": 24, "PB": 25, "PE": 26, "AL": 27,
    "SE": 28, "BA": 29, "MG": 31, "ES": 32, "RJ": 33, "SP": 35, "PR": 41,
    "SC": 42, "RS": 43, "MS": 50, "MT": 51, "GO": 52, "DF": 53,
}

CODE_TO_UF = {code: abbr for abbr, code in UF_ABBR_TO_CODE.items()}

# PDF 1 = data/dic_dados_dengue.pdf
# PDF 2 = data/dic_dados_dengue_alternative.pdf

DEFAULT_NULL_DICT = {
    # Sex: "I" = Ignorado. PDF 1 f.14 (cs_sexo). Not documented in PDF 2.
    ("CS_SEXO",): ["I"],

    # 9 = Ignorado.
    #   CS_GESTANT  PDF 1 f.15   (see note on code 4 below)
    #   CS_RACA     PDF 1 f.16
    #   HOSPITALIZ  PDF 1 f.62 / PDF 2 f.50   (1-Sim 2-Não 9-Ignorado)
    #   DOENCA_TRA  PDF 1 f.72                (1-Sim 2-Não 9-Ignorado)
    #   EVOLUCAO    PDF 1 f.74 / PDF 2 f.65   (...9-ignorado)
    #   Hemorrhagic manifestations block, PDF 1 f.103-115, all "1-Sim 2-Não 9-Ignorado".
    #   This whole block is NOT in PDF 2 (Sinan Online dropped it), so it exists
    #   mainly in older/Net years. Present-columns filter handles its absence.
    ("CS_GESTANT", "CS_RACA", "HOSPITALIZ", "DOENCA_TRA", "EVOLUCAO",
     "MANI_HEMOR", "EPISTAXE", "GENGIVO", "METRO", "PETEQUIAS", "HEMATURA",
     "SANGRAM", "LACO_N", "PLASMATICO", "EVIDENCIA", "CON_FHD", "COMPLICA"): [9],

    # Education: Ign/Branco = 9, 09, 99. PDF 1 f.17 (cs_escol_n). Not in PDF 2.
    ("CS_ESCOL_N",): [9, 99],

    # Lab results: 4 = "Não realizado".
    #   RES_CHIKS1/2, RESUL_PRNT   PDF 1 f.48,49,50 / PDF 2 f.38  (1-Reagente 2-Não Reagente 3-Inconclusivo 4-Não realizado)
    #   RESUL_SORO                 PDF 1 f.52 / PDF 2 f.40
    #   RESUL_NS1, RESUL_VI_N, RESUL_PCR_, HISTOPA_N, IMUNOH_N
    #                              PDF 1 f.54,56,58,60,61 / PDF 2 f.42,43,46,48,49
    #                              (PDF 2 relabels these 1-Positivo 2-Negativo ... but 4 is still "Não realizado")
    ("RES_CHIKS1", "RES_CHIKS2", "RESUL_PRNT", "RESUL_SORO", "RESUL_NS1",
     "RESUL_VI_N", "RESUL_PCR_", "HISTOPA_N", "IMUNOH_N"): [4],

    # Classification & criterion: 0 and 9 = Ignorado.
    #   CLASSI_FIN  PDF 1 f.70  ("' ', 0, 9 Ignorado; 1 Confirmado, 2 Descartado, 8 Inconclusivo")
    #   CRITERIO    PDF 1 f.71  ("' ', 0, 9 Ignorado; 1,2,3")
    #   PDF 2 (f.62, f.63) lists no 0/9 code at all, so these appear in older years.
    #   8 (Inconclusivo) and the Online codes 5/10/11/12/13 are real categories, not null.

    # Assumed CON_CLASSI and CON_CRITER, the older versions of CLASSI_FIN and CRITERIO, have similar encodings because 
    # there's no evidence either way..
    ("CLASSI_FIN", "CON_CLASSI", "CRITERIO", "CON_CRITER"): [0, 9],
}

# full years being investigated, to be used by default unless a subset is required
DEFAULT_YEARS = [str(y) for y in np.arange(2001, 2020, 1)]

# first year of each new SINAN system: SINAN Windows -> SINAN Net in 2007, SINAN Net -> SINAN Online in 2014
SYSTEM_CHANGE_YEARS = [2007, 2014]

# states with decent rain-gauge coverage, chosen in 02-UNIPLU_Exploratory.ipynb (PR is later dropped from the models)
FOCUS_STATES = ["CE", "PB", "SP", "PR", "RJ"]

# last year of the training data; later years are the test set
TRAIN_END_YEAR = 2015


def load_dengue_data(data_dir=SINAN_DIR, years_to_read=DEFAULT_YEARS):
    """Scan the yearly SINAN dengue CSVs in data_dir (eg DENGBR01.csv) into a {year: LazyFrame} dict,
    keeping only the years in years_to_read."""
    data_dir = Path.Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"SINAN dengue data not found: {data_dir} does not exist. Download the dataset from {SINAN_URL} "
            f"and store the yearly CSV files (DENGBR00.csv to DENGBR25.csv) in the data folder, in a subfolder "
            f"called '{data_dir.name}' (ie {data_dir})."
        )
    lf_dict = dict()
    for item in data_dir.glob("*.csv"):
        year_str = str(2000 + int(item.stem[-2:])) # filenames are eg, DENGBR01
        if year_str not in years_to_read:
            continue
        else:
            lf_dict[year_str] = pl.scan_csv(
                    item,
                    schema_overrides={"NDUPLIC": pl.String, "NDUPLIC_N": pl.Int64},
                    ignore_errors=True,
                    infer_schema_length=10000,
                )
    missing = sorted(set(years_to_read) - set(lf_dict))
    if missing:
        raise FileNotFoundError(
            f"No SINAN dengue file for the year(s) {', '.join(missing)} in {data_dir}. Download the dataset from "
            f"{SINAN_URL} and store all the yearly CSV files (eg DENGBR01.csv for 2001) in that folder."
        )
    return lf_dict


def load_municipality_codes(csv_path=MUNICIPALITY_CSV):
    """IBGE municipality codes, names and states (UF) from the Receita Federal municipality table,
    with the accents in the names corrected."""
    csv_path = Path.Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Municipality table not found: {csv_path} does not exist. It is a copy of the Receita Federal "
            f"'Tabela de Municipios' ({MUNICIPALITY_URL}) with the accents in the names restored, and is part of "
            f"the repository: restore it in the data folder as '{csv_path.name}'."
        )
    return pl.read_csv(csv_path, separator=";").select("CÓDIGO DO MUNICÍPIO - IBGE", "MUNICÍPIO - IBGE", "UF")

def column_presence(lf_dict):
    """Build a year x column presence matrix from a {year: LazyFrame} dict.

    Returns ``(wide, fully_present)``:
      - ``wide``: one row per year,  plus a 0/1 column per column name seen in any file.
      - ``fully_present``: sorted list of columns present in *every* year.
    """
    colnames = {yr: lf.collect_schema().names() for yr, lf in lf_dict.items()}
    years = sorted(colnames)
    unique_colnames = sorted(set().union(*colnames.values()))
    set_dict = {yr: set(cols) for yr, cols in colnames.items()}

    wide = pl.DataFrame(
        {"year": years}
        | {col: [int(col in set_dict[yr]) for yr in years] for col in unique_colnames}
    )
    fully_present = sorted(set.intersection(*set_dict.values()))
    return wide, fully_present


def transform_nulls_dict(lf, null_dict):
    """Turn encoded null values (0 and 9, i.e. 'ignorado') into real nulls, based on an input
    dictionary of column names and their null encoding. If a column is not present in the 
    lazyframe, it is ignored."""

    present = set(lf.collect_schema().names())
    exprs = []
    for cols, null_vals in null_dict.items():
        cols_here = [c for c in cols if c in present] # to maintain order
        if not cols_here:
            continue
        numeric = all(isinstance(v, (int, float)) for v in null_vals)
        key = pl.col(*cols_here).cast(pl.Float32 if numeric else pl.String, strict=False)
        exprs.append(
            pl.when(key.is_in(null_vals))
              .then(None)
              .otherwise(pl.col(*cols_here))
              .name.keep()
        )
    return lf.with_columns(exprs)


def pct_in_values(column, values):
    """Expression: percentage of rows where ``column`` is in ``values``.

    The mean of a boolean column x 100 (the mean of True/False is the fraction of True), so no separate
    row count is needed. is_in gives null (not False) for rows where ``column`` is null, and mean() skips
    nulls, so fill_null(False) keeps those rows in the denominator as 'not in values'; without it the
    percentage would only cover the non-null rows.
    """
    return pl.col(column).is_in(values).fill_null(False).mean() * 100


def check_coverage(lf, cols_of_interest, nulls_dict=None, count_column=None, count_values=None,
                   count_alias="pct_in_values"):
    """Per-column presence flags and null percentages for one lazy frame, optionally also the
    percentage of rows where ``count_column`` is in ``count_values``.

    nulls_dict maps tuples of column names to their null encodings, eg ("COL1", "COL2"): [0, 9]
    when 0 and 9 mean 'ignorado' in those columns. These codes are turned into real nulls first, so
    they count as missing. Columns in nulls_dict that aren't in lf are skipped. None: no recoding.

    The null percentage is the mean of is_null() x 100: the mean of a boolean column is the fraction
    of True values, so no separate row count is needed.

    count_column / count_values / count_alias: if given, an extra column ``count_alias`` holds
    pct_in_values(count_column, count_values). It is computed after the null recoding, so count_values
    should not include null codes. If count_column is not in lf (eg CLASSI_FIN before 2007) the value is
    null, meaning 'not available'; deliberately not 0 (which would mean 'none') or -1 (used below).

    Returns ``(presence, nulls)`` as one-row lazy frames, with the columns in alphabetical order. Columns
    absent from ``lf`` get presence 0.0 and a null percentage of -1.0 (sentinel value).
    """
    cols_of_interest = sorted(cols_of_interest)
    lf_cols = lf.collect_schema().names()
    present_cols = [col for col in cols_of_interest if col in lf_cols]
    if nulls_dict:
        # transform_nulls_dict skips columns that aren't in this year's file
        lf = transform_nulls_dict(lf, nulls_dict)

    presence = pl.LazyFrame({col: [float(col in lf_cols)] for col in cols_of_interest})
    exprs = [
        (pl.col(col).is_null().mean() * 100).alias(col)
        if col in present_cols
        else pl.lit(-1.0).alias(col)
        for col in cols_of_interest
    ]
    if count_column is not None:
        exprs.append(
            pct_in_values(count_column, count_values).alias(count_alias)
            if count_column in lf_cols
            else pl.lit(None, dtype=pl.Float64).alias(count_alias)
        )
    nulls = lf.select(exprs)
    return presence, nulls


def get_confirmed_cases(lf, year):
    """Keep only confirmed dengue cases, using the classification column of the year's system: CON_CLASSI before
    the 2007 SINAN change, CLASSI_FIN from 2007. Both are assumed to use the same codes (DENGUE_CONFIRMADO =
    confirmed), which is supported by a similar share of confirmed notifications in both periods."""
    class_col = "CON_CLASSI" if int(year) < 2007 else "CLASSI_FIN"
    return lf.filter(pl.col(class_col).is_in(DENGUE_CONFIRMADO))


def clean_bad_cases(lf, year):
    """Drop duplicates, non-individual notifications and non-Brazil cases.

    The country code for Brazil changed with the 2007 SINAN system switch
    (800 -> 1). Espirito Santo (UF 32) is also excluded due to its internal
    system change. Pre-2007 SG_UF_NOT abbreviations are converted to IBGE codes
    so the column is numeric in every year.
    """
    if int(year) < 2007:
        lf = lf.filter(pl.col("NDUPLIC").ne_missing("S"))  # assume S = "sim" = duplicate
        # abbreviation -> IBGE code (e.g. "MG" -> 31); unknown values become null
        lf = lf.with_columns(
            pl.col("SG_UF_NOT").replace_strict(UF_ABBR_TO_CODE, default=None, return_dtype=pl.Int64)
        )
        brazil_code = 800  # pre-2007 SINAN Windows codes Brazil as 800
    else:
        # non-duplicate cases and only individual notifications
        lf = lf.filter(pl.col("NDUPLIC_N").ne_missing(2), pl.col("TP_NOT") == 2)
        brazil_code = 1  # 2007+ SINAN codes Brazil as 1

    return lf.filter(pl.col("ID_PAIS") == brazil_code, pl.col("SG_UF_NOT") != 32)

def convert_dates(lf):
    """
    Convert any columns starting with "DT_" to date format 
    """
    return lf.with_columns((cs.matches(r"^DT_") & cs.string()).str.to_date(strict=False))


def _fill_year_gaps(pop):
    """Fill missing (state, year) rows by linear interpolation over year within each state.

    The IBGE estimate table has no figures for the population count and census years (2007 and 2010),
    which would otherwise leave those years without a population denominator.
    """
    years = range(pop["year"].min(), pop["year"].max() + 1)
    states = pop["SG_UF_NOT"].unique().to_list()
    grid = pl.DataFrame(
        {"SG_UF_NOT": [s for s in states for _ in years],
         "year": [y for _ in states for y in years]}
    )
    return (
        grid.join(pop, on=["SG_UF_NOT", "year"], how="left")
            .sort("SG_UF_NOT", "year")
            .with_columns(pl.col("population").interpolate().over("SG_UF_NOT"))
            .with_columns(pl.col("population").round().cast(pl.Int64))
    )


def fetch_population(csv_path=POPULATION_CSV, cache=True):
    """Download IBGE state population estimates (SIDRA table 6579) and optionally store in csv_path.
    Missing census-year figures are interpolated so every year has a denominator.
    """
    url = "https://apisidra.ibge.gov.br/values/t/6579/n3/all/v/all/p/all"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)

    pop = pl.DataFrame(data[1:]).select(  # data[0] is the header row of labels, drop it
        SG_UF_NOT=pl.col("D1C").cast(pl.Int64),
        year=pl.col("D3C").cast(pl.Int64),
        population=pl.col("V").cast(pl.Int64),
    )
    pop = _fill_year_gaps(pop)
    if cache is True:
        pop.write_csv(csv_path)
    return pop

def cases_per_capita(lf, pop_lf, year, per=100_000, states=None):
    """Calculates cases per `per` inhabitants, per state, for one year.

    states: None -> all states; or a list of UF codes, e.g. [35, 33, 31].
    Accepts a DataFrame or LazyFrame for `lf`; always returns a LazyFrame.
    """
    if states:
        lf = lf.filter(pl.col("SG_UF_NOT").is_in(states))

    return (
        lf.filter(pl.col("DT_NOTIFIC").dt.year() == int(year))
          .group_by("SG_UF_NOT")
          .agg(pl.len().alias("cases")) # count number of rows = num cases
          .join(
              pop_lf.filter(pl.col("year") == int(year)).select("SG_UF_NOT", "population"),
              on="SG_UF_NOT", how="left",
          )
          .with_columns((pl.col("cases") / pl.col("population") * per).alias("rate"))
          .sort("SG_UF_NOT")
    )


def cases_per_capita_monthly(lf, pop_lf, year, per=100_000, states=None):
    """Calculates cases per `per` inhabitants, per state and month, for one year.

    Months with no notified cases are included with cases = 0.
    The denominator is the state's population for that year, so rates are
    per month (not annualised).
    states: None -> all states; or a list of UF codes, e.g. [35, 33, 31].
    Accepts a DataFrame or LazyFrame for `lf`; always returns a LazyFrame.
    """
    year = int(year)
    pop = pop_lf.filter(pl.col("year") == year).select("SG_UF_NOT", "population")
    if states:
        lf = lf.filter(pl.col("SG_UF_NOT").is_in(states))
        pop = pop.filter(pl.col("SG_UF_NOT").is_in(states))

    months = pl.LazyFrame({
        "month": pl.date_range(date(year, 1, 1), date(year, 12, 1), interval="1mo", eager=True)
    })

    monthly_cases = (
        lf.filter(pl.col("DT_NOTIFIC").dt.year() == year)
          .with_columns(pl.col("DT_NOTIFIC").dt.truncate("1mo").cast(pl.Date).alias("month"))
          .group_by("SG_UF_NOT", "month")
          .agg(pl.len().alias("cases"))
    )

    return (
        pop.join(months, how="cross")
           .join(monthly_cases, on=["SG_UF_NOT", "month"], how="left")
           .with_columns(pl.col("cases").fill_null(0))
           .with_columns((pl.col("cases") / pl.col("population") * per).alias("rate"))
           .sort("SG_UF_NOT", "month")
    )

def load_population(csv_path=POPULATION_CSV):
    """Load the cached population CSV as a lazy frame, downloading it first (see :func:`fetch_population`)
    if it doesn't exist yet."""
    if not Path.Path(csv_path).is_file():
        fetch_population(csv_path)
    return pl.scan_csv(csv_path)

def load_rainfall_spatiotemporal(parquet_path=RAINFALL_CLEANED_DIR / "full_spatiotemporal", years=DEFAULT_YEARS):
    """Scan the daily gauge readings with their health region and state (saved by 02-UNIPLU_Exploratory.ipynb)
    into a {year: LazyFrame} dict."""
    spatiotemporal_rainfall_path = Path.Path(parquet_path)
    if not spatiotemporal_rainfall_path.is_dir():
        raise FileNotFoundError(
            f"Processed rainfall data not found in {spatiotemporal_rainfall_path}. It is made by "
            f"02-UNIPLU_Exploratory.ipynb from the UNIPLU-BR data ({UNIPLU_URL}), so run that notebook first."
        )
    spatiotemporal_rain_data = dict()
    for year in years:
        spatiotemporal_rain_data[str(year)] = pl.scan_parquet(Path.Path(spatiotemporal_rainfall_path / f"{year}_rain_with_region.parquet"))

    return spatiotemporal_rain_data

def full_basic_clean(lf_dict, drop_nulls=False, select_cols = [], null_dict=DEFAULT_NULL_DICT):
    """Performs very basic cleaning on the given lazyframe dictionary, returning a dictionary
    of cleaned LFs. Gets the confirmed cases, drops obviously irrelevant cases (eg those outside Brazil),
    converts dates to date type, transforms null codes into proper nulls according to 'null_dict'.
    
    By default, null values are not dropped and no specific columns are selected, but this is optionally
    available to reduce the number of times one has to loop over the lazyframe dictionary to get workable data."""
    clean_dict = dict()
    for yr_str, lf in lf_dict.items():
        cleaned_lf = (lf.pipe(get_confirmed_cases, yr_str)
            .pipe(clean_bad_cases, yr_str)
            .pipe(convert_dates)
            .pipe(transform_nulls_dict, null_dict))

        if len(select_cols) > 0:
            cleaned_lf = cleaned_lf.select(select_cols)

        if drop_nulls == True:
            cleaned_lf = cleaned_lf.drop_nulls()

        clean_dict[yr_str] = cleaned_lf
    return clean_dict
