# Dengue Prediction

This project looks at how well monthly dengue rates in Brazil can be explained and predicted from past dengue cases, the time of year and rainfall. It combines three data sources:

- **Dengue notifications** from SINAN, Brazil's notifiable diseases information system (2001-2019).
- **Population estimates** from IBGE, to turn case counts into rates (cases per 100,000 inhabitants).
- **Daily rainfall** from rain the Hidroweb gauges daiy readings in the UNIPLU-BR dataset.

The analysis focuses on four states with good rain data coverage: Ceará (CE), Paraíba (PB), São Paulo (SP) and Rio de Janeiro (RJ). For each state:

- A linear regression of log(monthly rate) on the rate in a previous month and seasonal (sine/cosine) terms is trained on 2001-2015 and tested on 2016-2019, against a naive baseline that predicts the rate from a previous month. Lagged rainfall is then added to see whether it improves on the seasonal terms.
- An STL decomposition splits each state's rates into trend, seasonal pattern and remainder, to describe the normal dengue season and to spot unusual years.

## Repository structure

```
Dengue Prediction/
├── data/                  raw data (the large datasets are not in the repository, see below)
├── output/                everything the code writes (tables, cached population data, processed rainfall)
└── src/
    ├── 01-SINAN_Exploratory.ipynb    exploration and cleaning of the SINAN dengue data
    ├── 02-UNIPLU_Exploratory.ipynb   exploration of the rain gauges, choice of states, processed rainfall
    ├── 03-Models.ipynb               regression models and STL decomposition
    ├── pipeline_funcs.py             data loading/cleaning helpers and shared settings (paths, years, states)
    └── visualize.py                  plotting helpers
```

## Data

Two datasets are too large for the repository and have to be downloaded by hand into the `data` folder, with exactly these folder names:

```
data/
├── SINAN_data/          DENGBR00.csv, DENGBR01.csv, ..., DENGBR25.csv
├── UNIPLU_data/         AC_1961.zip, ..., SP_2010.zip, ..., TO_2025.zip
├── Tabela_de_Municipios_corrigida.csv
├── Tabela de Municipios.csv
├── dic_dados_dengue.pdf
└── dic_dados_dengue_alternative.pdf
```

- **SINAN dengue notifications** (`data/SINAN_data`): download the dataset from Kaggle, [Dengue no Brasil (SINAN) 2000-2025](https://www.kaggle.com/datasets/gallo33henrique/dengue-no-brasil-sinan-20002025), and put the yearly CSV files (`DENGBR00.csv` for 2000 up to `DENGBR25.csv` for 2025) directly in `data/SINAN_data`. The code uses 2001-2019, so at least `DENGBR01.csv` to `DENGBR19.csv` are needed. The two `dic_dados_dengue` PDFs in `data` are the data dictionaries describing the columns and their codes.
- **UNIPLU-BR rain gauge data** (`data/UNIPLU_data`): download the dataset from Zenodo, [Unified Brazilian Rainfall Dataset (UNIPLU-BR)](https://doi.org/10.5281/zenodo.18883358) (a single zip file of about 1.6 GB), unzip it and put the zip files per state and year (named `{STATE}_{YEAR}.zip`, eg `SP_2010.zip`, each containing `table_data.parquet` and `table_info.parquet`) directly in `data/UNIPLU_data`. Don't unzip these per-state files; the code reads them as they are. The scripts that come with the dataset are on [GitHub](https://github.com/LARHENA/UNIPLU-BR).
- **Municipality table** (`data/Tabela_de_Municipios_corrigida.csv`): this one is included in the repository. It is the Receita Federal's table of municipalities and their IBGE codes, available from [gov.br (Órgãos e Municípios)](https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/orgaos-e-municipios) ([direct CSV download](https://www.gov.br/receitafederal/dados/municipios.csv)). The downloaded file (`Tabela de Municipios.csv`) is missing the accented letters in the names (eg 'São João' becomes 'So Joo'), so the code uses `Tabela_de_Municipios_corrigida.csv`, a copy with the names corrected. It is provided for your convenience.

Two more sources are downloaded automatically, so they need an internet connection the first time:

- **Population estimates**: IBGE [SIDRA table 6579](https://sidra.ibge.gov.br/tabela/6579), fetched through the SIDRA API and cached in `output/pop_estimates.csv`. There are no estimates for 2007 and 2010 (the years of IBGE's population count and census), so these are interpolated linearly per state.
- **Maps** of Brazil's states, municipalities and health regions, from the [geobr](https://github.com/ipeaGIT/geobr) package (used in `02-UNIPLU_Exploratory.ipynb`).

## How to run

1. Install Python (the code was developed with Python 3.13) and the packages:

   ```bash
   pip install polars pyarrow numpy pandas matplotlib statsmodels scipy geopandas geobr shapely jupyter
   ```

   Versions used: polars 1.44, pyarrow 25.0, numpy 2.5, pandas 3.0, matplotlib 3.11, statsmodels 0.15, scipy 1.18, geopandas 1.2, geobr 2.1, shapely 2.2.

2. Download the data and store it in `data` as described above.

3. Open the notebooks in `src` (eg with `jupyter lab` from the repository folder) and run them from top to bottom, in this order:
   1. `01-SINAN_Exploratory.ipynb`: explores the dengue data and explains the cleaning choices. It writes summary tables of column presence, missing values and confirmed cases to `output`.
   2. `02-UNIPLU_Exploratory.ipynb`: explores the rain gauges, explains the choice of states and saves the processed rainfall data in `output/Rainfall_cleaned`. **Run this before `03-Models.ipynb`**, which reads that processed data. If `output/Rainfall_cleaned` already exists, the notebook reads the saved data instead of filtering and saving the raw UNIPLU-BR data again (delete that folder to recreate it). `03-Models.ipynb` only needs `output/Rainfall_cleaned`, not the raw UNIPLU-BR data.
   3. `03-Models.ipynb`: builds the monthly rates, fits and compares the regression models and runs the STL decomposition.

   All paths are set in `src/pipeline_funcs.py` relative to the repository folder, so the notebooks work wherever the repository is stored.

## Citations

- SINAN dengue notifications: gallo33henrique, *Dengue no Brasil (SINAN) 2000-2025*, Kaggle. https://www.kaggle.com/datasets/gallo33henrique/dengue-no-brasil-sinan-20002025
- UNIPLU-BR: Lemos, F. C., Freitas, E. S., Coelho, V. H. R., ... & Almeida, C. N. (2026). *Unified Brazilian Rainfall Dataset (UNIPLU-BR): A Standardized National Database of Point Precipitation from Major Brazilian Monitoring Networks* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18883358
- Municipality table: Receita Federal, *Tabela de Órgãos e Municípios*. https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/orgaos-e-municipios
- Population estimates: IBGE, SIDRA table 6579. https://sidra.ibge.gov.br/tabela/6579
- Seasonal decomposition method: Hyndman, R. J. & Athanasopoulos, G., *Forecasting: Principles and Practice* (3rd ed.), ch. 3. https://otexts.com/fpp3/decomposition.html
