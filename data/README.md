# Data

The paper's empirical results are based on data from Bloomberg Terminal. Bloomberg's license does not permit redistribution of raw data, so this folder is empty in the public repository. To reproduce the paper's numerical results, source the files yourself following the schemas below and place them in this directory.

## File 1: `USDJPY.csv`

5-minute bars of USD/JPY spot, January 2023 through the analysis date (or longer — the code handles any start date ≥ January 2023).

### Bloomberg sourcing

Any standard Bloomberg BDH export of 5-minute USD/JPY bars works. Common routes include the Bloomberg Excel Add-in's `BDH` formula with the `FREQ=5` parameter, or a direct API pull via `blpapi`. The CSV should have a header-style layout where the top few rows contain Bloomberg metadata (sometimes including the ticker, timezone info, etc.) and a row containing the word `Dates` marks the start of the tabular data.

### Required columns

After the `Dates` header row, the file must contain these columns in some order (the parser finds them by name where possible, but falls back to positional parsing if only five numeric columns are present):

| Column | Type    | Description                      |
|--------|---------|----------------------------------|
| Dates  | datetime | Bar timestamp. Excel serial format or ISO 8601 both accepted. |
| Open   | float   | Opening price of the 5-minute bar |
| Close  | float   | Closing price of the 5-minute bar |
| High   | float   | Highest price within the bar     |
| Low    | float   | Lowest price within the bar      |

### Timezone handling

If the timestamps are timezone-naive, the code assumes UTC and converts to Asia/Tokyo internally. If they are already timezone-aware, the code converts whatever timezone they carry to Asia/Tokyo. Either input works; the only constraint is that the timestamps must be convertible to Tokyo local time unambiguously.

### Defensive handling

The parser will:
- Swap `High` and `Low` if more than 10% of rows have `High < Low` (occasional Bloomberg column-order quirk)
- Drop rows where `Close` or `Dates` are null
- Handle Excel serial dates (numeric) by converting from the 1899-12-30 origin

You don't need to clean the file before providing it — just export and drop it in.

## File 2: `JPEquities.xlsx`

Daily OHLC data for the three firms studied in the paper: Sumitomo Metal Mining (ticker 5713), Mitsubishi Materials (5711), and Mitsui Mining & Smelting (5706).

### Structure

One sheet per ticker. The parser identifies the correct sheet by checking:
1. Sheet name contains the ticker code (e.g., a sheet named `5713` or `5713 JT Equity` matches ticker 5713).
2. If no sheet name matches, the parser searches the first 15 rows × 5 columns of each sheet for the ticker string.

### Required columns per sheet

| Column   | Type     | Description                          |
|----------|----------|--------------------------------------|
| Dates    | datetime | Trading date. Excel serial or ISO.   |
| PX_OPEN  | float    | Daily opening price                  |
| PX_LAST  | float    | Daily closing price                  |

Additional columns are tolerated and ignored. The parser looks for a header row containing `Dates` or `PX_LAST` within the first 20 rows of each sheet, which handles Bloomberg's typical BDH output format where metadata occupies the top of the sheet.

### Bloomberg sourcing

The standard Bloomberg BDH pull for each ticker with fields `PX_OPEN` and `PX_LAST` produces the expected output. Tickers to query: `5706 JT Equity`, `5711 JT Equity`, `5713 JT Equity`. The timespan should match the FX data's window.

### Timezone handling

Equity timestamps don't need timezone information — they are treated as calendar dates corresponding to Tokyo trading days. The merge with FX data happens on the `Pred_Date` column, which is derived from the FX overnight window (15:00 prior day → 08:50 current day Tokyo time) and maps to the next Tokyo business day.

## Data integrity checks the code performs

When you run the pipeline, the following are verified at runtime:

1. **Timestamp-level lookahead guard.** No FX bar with a timestamp at or after 09:00 JST on the prediction date can enter the feature set. If any bar violates this, the pipeline raises `ValueError` with the offending timestamp. This is a hard error, not a warning; the analysis cannot proceed with a violation present.

2. **Minimum observations per day.** Days with fewer than 50 valid FX bars in the overnight window are dropped. This removes holidays, partial trading sessions, and data-quality outages.

3. **Gap truncation.** Opening gaps with absolute value below 10^-5 are truncated to zero to mitigate Bloomberg rounding artifacts. See `CODE_REVIEW_NOTES.md` for discussion.

4. **Embargo at train/test boundary.** The chronological 80/20 split enforces a two-business-day gap between the last training observation and the first test observation to eliminate short-term autocorrelation leakage.

## What happens if data doesn't match the schema

The loading code is defensive but not silent. If the schema is violated — missing columns, unparseable dates, empty sheets — you'll get one of:

- `None` return from `DataEngine.get_data(ticker)`, followed by a warning that the experiment is skipped.
- `ValueError` with a specific message if the lookahead guard is violated.
- Standard Python exceptions from `pandas` or `statsmodels` downstream if data types are wrong.

The errors are informative. Read them; they usually point to the specific column or row causing the problem.

## File placement

Place `USDJPY.csv` and `JPEquities.xlsx` directly in this `data/` directory. The default `ResearchConfig.base_path` resolves to `<repo_root>/data/`, so no additional configuration is needed. Alternatively, set `FX_GAP_DATA_DIR` to any directory containing these two files, or pass `--data-dir <path>` on the command line.

## Data lineage note

Because the raw data is not in the repository, readers cannot independently verify that the code's numerical outputs match the paper's reported numbers without sourcing the data themselves. This is an irreducible limitation of any paper using proprietary data sources. Readers who want to verify the pipeline's correctness without Bloomberg access can still: inspect the code, run the unit tests (which use hand-constructed fixtures rather than real data), and verify that the lookahead guard and schema validation behave as documented.
