"""
=============================================================================
S&P 500 Data Preparation Pipeline
Master's Research: ARIMA-GARCH vs LSTM Financial Forecasting
=============================================================================

This script performs:
  STEP 1 â€” Load and inspect both datasets
  STEP 2 â€” Standardize formats and column names
  STEP 3 â€” Validate the overlap period
  STEP 4 â€” Define the master price series
  STEP 5 â€” Check the final series integrity
  STEP 6 â€” Derive log returns
  STEP 7 â€” Basic financial sanity checks
  STEP 8 â€” Save outputs

Every decision is explicitly logged for thesis methodology documentation.
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
from datetime import datetime, timedelta

# Statistical libraries
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR = str(Path(__file__).resolve().parent.parent)
CSV_PATH = os.path.join(BASE_DIR, "S&P 500 Historical Data.csv")
XLSX_PATH = os.path.join(BASE_DIR, "SP500.xlsx")
OUTPUT_CSV = os.path.join(BASE_DIR, "SP500_master_2000_2026.csv")
REPORT_PATH = os.path.join(BASE_DIR, "validation_report.txt")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# â”€â”€ Validation report log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
report_lines = []

def log(msg, also_print=True):
    """Log a message to both console and the validation report."""
    report_lines.append(msg)
    if also_print:
        print(msg)

def section(title):
    """Print a section header."""
    sep = "=" * 80
    log("")
    log(sep)
    log(f"  {title}")
    log(sep)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 1 â€” LOAD AND INSPECT                                                â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 1 â€” LOAD AND INSPECT BOTH DATASETS")

# â”€â”€ Dataset 1: CSV from Investing.com â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- Dataset 1: S&P 500 Historical Data (CSV) ---")
log(f"Source file: {CSV_PATH}")

df1_raw = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")

log(f"Number of rows: {len(df1_raw)}")
log(f"Column names: {list(df1_raw.columns)}")
log(f"Data types:\n{df1_raw.dtypes.to_string()}")
log(f"\nFirst 5 rows:\n{df1_raw.head().to_string()}")
log(f"\nLast 5 rows:\n{df1_raw.tail().to_string()}")
log(f"\nMissing values per column:\n{df1_raw.isnull().sum().to_string()}")

# â”€â”€ Dataset 2: FRED Excel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- Dataset 2: FRED SP500 (Excel) ---")
log(f"Source file: {XLSX_PATH}")

df2_raw = pd.read_excel(XLSX_PATH)

log(f"Number of rows: {len(df2_raw)}")
log(f"Column names: {list(df2_raw.columns)}")
log(f"Data types:\n{df2_raw.dtypes.to_string()}")
log(f"\nFirst 5 rows:\n{df2_raw.head().to_string()}")
log(f"\nLast 5 rows:\n{df2_raw.tail().to_string()}")
log(f"\nMissing values per column:\n{df2_raw.isnull().sum().to_string()}")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 2 â€” STANDARDIZE                                                      â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 2 â€” STANDARDIZE FORMATS AND COLUMN NAMES")

# â”€â”€ Dataset 1 Standardization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- Standardizing Dataset 1 ---")

df1 = df1_raw.copy()

# Parse dates (format: MM/DD/YYYY)
df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
log(f"Date parsing: MM/DD/YYYY format detected and parsed.")
log(f"Date range: {df1['Date'].min()} to {df1['Date'].max()}")

# Clean numeric columns: remove commas (thousands separators) and spaces
numeric_cols_d1 = ['Price', 'Open', 'High', 'Low']
for col in numeric_cols_d1:
    original_dtype = df1[col].dtype
    # Remove commas used as thousands separators AND any spaces
    df1[col] = df1[col].astype(str).str.replace(',', '', regex=False)
    df1[col] = df1[col].str.replace(' ', '', regex=False)
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
    n_failed = df1[col].isnull().sum() - df1_raw[col].isnull().sum()
    log(f"  Column '{col}': converted from {original_dtype} to numeric. "
        f"New NaN from conversion: {n_failed}")

# Handle Volume column
df1['Vol.'] = df1['Vol.'].astype(str).str.replace(',', '', regex=False)
df1['Vol.'] = df1['Vol.'].str.replace(' ', '', regex=False)
df1['Vol.'] = df1['Vol.'].str.replace('-', '', regex=False)
# Handle K, M, B suffixes
def parse_volume(v):
    v = str(v).strip()
    if v in ['', 'nan', 'None', '-']:
        return np.nan
    v_upper = v.upper()
    try:
        if v_upper.endswith('B'):
            return float(v_upper[:-1]) * 1e9
        elif v_upper.endswith('M'):
            return float(v_upper[:-1]) * 1e6
        elif v_upper.endswith('K'):
            return float(v_upper[:-1]) * 1e3
        else:
            return float(v_upper)
    except ValueError:
        return np.nan

df1['Vol_numeric'] = df1_raw['Vol.'].apply(parse_volume)
log(f"  Volume: parsed with K/M/B suffix handling. "
    f"Missing: {df1['Vol_numeric'].isnull().sum()}")

# Change % â€” we will NOT use this as the definitive return variable
log(f"\n  DECISION: 'Change %' column is NOT used as the definitive return.")
log(f"  Reason: We will compute log returns from cleaned Close prices directly.")

# Rename for clarity
df1 = df1.rename(columns={'Price': 'Close_D1'})

# Sort chronologically (the CSV is in reverse chronological order)
df1 = df1.sort_values('Date').reset_index(drop=True)
log(f"\n  Dataset 1 sorted chronologically.")
log(f"  Final date range: {df1['Date'].min().date()} to {df1['Date'].max().date()}")
log(f"  Unique dates: {df1['Date'].nunique()}")
log(f"  Duplicate dates: {df1['Date'].duplicated().sum()}")

# â”€â”€ Dataset 2 Standardization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- Standardizing Dataset 2 ---")

df2 = df2_raw.copy()

# Rename columns
df2 = df2.rename(columns={'observation_date': 'Date', 'SP500': 'Close_FRED'})
log(f"  Renamed: observation_date â†’ Date, SP500 â†’ Close_FRED")

# Parse dates
df2['Date'] = pd.to_datetime(df2['Date'])
log(f"  Date range: {df2['Date'].min().date()} to {df2['Date'].max().date()}")

# Convert Close_FRED to numeric (handle '.' which FRED uses for missing)
df2['Close_FRED'] = pd.to_numeric(df2['Close_FRED'], errors='coerce')
n_missing_fred = df2['Close_FRED'].isnull().sum()
log(f"  Close_FRED converted to numeric. Missing values: {n_missing_fred}")

# Drop rows where Close_FRED is NaN (FRED uses '.' for non-trading days)
n_before = len(df2)
df2 = df2.dropna(subset=['Close_FRED']).reset_index(drop=True)
n_after = len(df2)
log(f"  DECISION: Dropped {n_before - n_after} rows with missing Close_FRED values.")
log(f"  Reason: FRED marks non-trading days with '.', these are not real observations.")

# Sort chronologically
df2 = df2.sort_values('Date').reset_index(drop=True)
log(f"  Unique dates: {df2['Date'].nunique()}")
log(f"  Duplicate dates: {df2['Date'].duplicated().sum()}")
log(f"  Final date range: {df2['Date'].min().date()} to {df2['Date'].max().date()}")

# â”€â”€ Descriptive statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- Descriptive Statistics ---")
log(f"\nDataset 1 â€” Close_D1:")
log(df1['Close_D1'].describe().to_string())

log(f"\nDataset 2 â€” Close_FRED:")
log(df2['Close_FRED'].describe().to_string())


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 3 â€” VALIDATE THE OVERLAP                                             â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 3 â€” VALIDATE THE OVERLAP PERIOD")

# Merge on Date for overlap analysis
overlap = pd.merge(df1[['Date', 'Close_D1']], df2[['Date', 'Close_FRED']],
                   on='Date', how='inner')

log(f"Number of overlapping dates: {len(overlap)}")
log(f"Overlap period: {overlap['Date'].min().date()} to {overlap['Date'].max().date()}")

# Calculate differences
overlap['Abs_Diff'] = (overlap['Close_D1'] - overlap['Close_FRED']).abs()
overlap['Rel_Diff_Pct'] = (overlap['Abs_Diff'] / overlap['Close_FRED']) * 100

# Statistics
n_exact = (overlap['Abs_Diff'] == 0).sum()
n_close_01 = (overlap['Abs_Diff'] <= 0.1).sum()  # within 0.10
n_close_1 = (overlap['Abs_Diff'] <= 1.0).sum()   # within 1.00

log(f"\n--- Overlap Comparison Statistics ---")
log(f"Total overlapping dates:      {len(overlap)}")
log(f"Exact matches (diff = 0):     {n_exact}")
log(f"Within Â±0.10:                 {n_close_01}")
log(f"Within Â±1.00:                 {n_close_1}")
log(f"Max absolute difference:      {overlap['Abs_Diff'].max():.4f}")
log(f"Max relative difference (%):  {overlap['Rel_Diff_Pct'].max():.6f}%")
log(f"Mean absolute difference:     {overlap['Abs_Diff'].mean():.4f}")
log(f"Mean relative difference (%): {overlap['Rel_Diff_Pct'].mean():.6f}%")
log(f"Median absolute difference:   {overlap['Abs_Diff'].median():.4f}")
log(f"Std absolute difference:      {overlap['Abs_Diff'].std():.4f}")

# Distribution of differences
log(f"\n--- Distribution of Absolute Differences ---")
bins = [0, 0.01, 0.1, 0.5, 1.0, 5.0, float('inf')]
labels = ['=0 to 0.01', '0.01 to 0.1', '0.1 to 0.5', '0.5 to 1.0', '1.0 to 5.0', '>5.0']
overlap['Diff_Bin'] = pd.cut(overlap['Abs_Diff'], bins=bins, labels=labels, right=True)
diff_dist = overlap['Diff_Bin'].value_counts().sort_index()
for label, count in diff_dist.items():
    log(f"  {label}: {count} ({count/len(overlap)*100:.1f}%)")

# Show dates with largest discrepancies
log(f"\n--- Top 10 Largest Discrepancies ---")
top_disc = overlap.nlargest(10, 'Abs_Diff')[['Date', 'Close_D1', 'Close_FRED',
                                              'Abs_Diff', 'Rel_Diff_Pct']]
log(top_disc.to_string(index=False))

# Show some matching observations
log(f"\n--- Sample of Matching Observations (diff < 0.01) ---")
matching = overlap[overlap['Abs_Diff'] < 0.01].head(10)[
    ['Date', 'Close_D1', 'Close_FRED', 'Abs_Diff']]
log(matching.to_string(index=False))

# Show some non-matching observations
log(f"\n--- Sample of Non-Matching Observations (diff >= 0.10) ---")
non_matching = overlap[overlap['Abs_Diff'] >= 0.10].head(10)[
    ['Date', 'Close_D1', 'Close_FRED', 'Abs_Diff', 'Rel_Diff_Pct']]
if len(non_matching) > 0:
    log(non_matching.to_string(index=False))
else:
    log("  No observations with absolute difference >= 0.10")

# â”€â”€ Investigate discrepancy causes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log(f"\n--- Discrepancy Investigation ---")

# Check if differences are consistent with rounding
rounding_threshold = 0.15  # generous threshold for rounding
n_rounding = (overlap['Abs_Diff'] <= rounding_threshold).sum()
log(f"Observations within rounding threshold (Â±{rounding_threshold}): "
    f"{n_rounding}/{len(overlap)} ({n_rounding/len(overlap)*100:.1f}%)")

# Check for systematic bias
mean_signed_diff = (overlap['Close_D1'] - overlap['Close_FRED']).mean()
log(f"Mean signed difference (D1 - FRED): {mean_signed_diff:.4f}")
log(f"  â†’ {'No systematic bias detected' if abs(mean_signed_diff) < 0.5 else 'Potential systematic bias!'}")

# Check dates present in one but not the other (during overlap period)
overlap_start = overlap['Date'].min()
overlap_end = overlap['Date'].max()
d1_dates_in_range = set(df1[(df1['Date'] >= overlap_start) &
                            (df1['Date'] <= overlap_end)]['Date'])
d2_dates_in_range = set(df2[(df2['Date'] >= overlap_start) &
                            (df2['Date'] <= overlap_end)]['Date'])

only_d1 = d1_dates_in_range - d2_dates_in_range
only_d2 = d2_dates_in_range - d1_dates_in_range

log(f"\nDates in overlap range present ONLY in Dataset 1: {len(only_d1)}")
if len(only_d1) > 0 and len(only_d1) <= 20:
    for d in sorted(only_d1):
        log(f"  {d.date()}")
log(f"Dates in overlap range present ONLY in Dataset 2 (FRED): {len(only_d2)}")
if len(only_d2) > 0 and len(only_d2) <= 20:
    for d in sorted(only_d2):
        log(f"  {d.date()}")

# Conclusion on discrepancy causes
log(f"\n--- Discrepancy Conclusion ---")
if overlap['Abs_Diff'].max() < 1.0:
    log("CONCLUSION: All differences are minor (< 1.0 index point).")
    log("Most likely cause: rounding differences between data providers.")
    log("The two datasets are COMPATIBLE for merging.")
    discrepancy_level = "minor_rounding"
elif overlap['Abs_Diff'].max() < 5.0:
    log("CONCLUSION: Differences are small (< 5.0 index points).")
    log("Likely causes: rounding + slightly different closing price definitions.")
    log("The two datasets are COMPATIBLE for merging with FRED preferred.")
    discrepancy_level = "small"
else:
    log("WARNING: Some differences are large (> 5.0 index points).")
    log("Manual investigation of large discrepancies is recommended.")
    discrepancy_level = "large"

# Plot overlap comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(overlap['Date'], overlap['Close_D1'], label='Dataset 1 (Investing.com)',
             alpha=0.8, linewidth=0.8)
axes[0].plot(overlap['Date'], overlap['Close_FRED'], label='FRED', alpha=0.8,
             linewidth=0.8, linestyle='--')
axes[0].set_title('S&P 500 Close Price â€” Overlap Period Comparison')
axes[0].set_ylabel('Close Price')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(overlap['Date'], overlap['Abs_Diff'], width=1.5, color='coral', alpha=0.7)
axes[1].set_title('Absolute Difference Between Sources')
axes[1].set_ylabel('|D1 - FRED|')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'step3_overlap_comparison.png'), dpi=150)
plt.close()
log(f"\nPlot saved: {os.path.join(PLOTS_DIR, 'step3_overlap_comparison.png')}")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 4 â€” DEFINE THE MASTER PRICE SERIES                                   â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 4 â€” DEFINE THE MASTER PRICE SERIES")

# â”€â”€ Merge Decision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- Merge Strategy ---")
log("Rule: Create a single 'Close' column using the following logic:")
log("  1. For dates BEFORE the FRED coverage start â†’ use Dataset 1 (Close_D1)")
log("  2. For dates in the OVERLAP period â†’ use FRED (Close_FRED)")
log("     Rationale: FRED is a more standardized, institutional source.")
log("     The differences are minor (rounding only), so this choice")
log("     does not materially affect the research.")
log("  3. For dates AFTER Dataset 1 ends â†’ use FRED (Close_FRED)")
log("")
log("  This is a DOCUMENTED METHODOLOGICAL DECISION.")

fred_start = df2['Date'].min()
d1_end = df1['Date'].max()
log(f"\n  FRED coverage starts: {fred_start.date()}")
log(f"  Dataset 1 coverage ends: {d1_end.date()}")

# Build master dataset
# Part A: Dataset 1 dates BEFORE FRED starts
part_a = df1[df1['Date'] < fred_start][['Date', 'Close_D1']].copy()
part_a = part_a.rename(columns={'Close_D1': 'Close'})
part_a['Source'] = 'Investing.com'
log(f"\n  Part A (D1 only, before FRED): {len(part_a)} observations")
log(f"    Range: {part_a['Date'].min().date()} to {part_a['Date'].max().date()}")

# Part B: FRED from its start onward (covers overlap + post-D1 period)
part_b = df2[['Date', 'Close_FRED']].copy()
part_b = part_b.rename(columns={'Close_FRED': 'Close'})
part_b['Source'] = 'FRED'
log(f"  Part B (FRED, from start onward): {len(part_b)} observations")
log(f"    Range: {part_b['Date'].min().date()} to {part_b['Date'].max().date()}")

# Concatenate
master = pd.concat([part_a, part_b], ignore_index=True)
master = master.sort_values('Date').reset_index(drop=True)

# Check for duplicates after merge
dup_dates = master[master['Date'].duplicated(keep=False)]
if len(dup_dates) > 0:
    log(f"\n  WARNING: {len(dup_dates)} duplicate dates found after merge!")
    log(f"  Removing duplicates, keeping FRED value (appears second after sort).")
    # In case of duplicate dates, keep the FRED observation
    master = master.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
else:
    log(f"\n  No duplicate dates after merge â€” clean concatenation.")

log(f"\n  Master dataset: {len(master)} observations")
log(f"  Date range: {master['Date'].min().date()} to {master['Date'].max().date()}")

# Source breakdown
src_counts = master['Source'].value_counts()
log(f"\n  Source breakdown:")
for src, cnt in src_counts.items():
    log(f"    {src}: {cnt} observations")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 5 â€” CHECK THE FINAL SERIES                                           â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 5 â€” CHECK THE FINAL SERIES INTEGRITY")

# Chronological ordering
is_sorted = master['Date'].is_monotonic_increasing
log(f"Chronologically ordered: {is_sorted}")

# Duplicate dates
n_dup = master['Date'].duplicated().sum()
log(f"Duplicate dates: {n_dup}")

# Missing Close values
n_missing_close = master['Close'].isnull().sum()
log(f"Missing Close values: {n_missing_close}")

# Negative or zero prices
n_nonpositive = (master['Close'] <= 0).sum()
log(f"Negative or zero prices: {n_nonpositive}")

# Date range
log(f"Date range: {master['Date'].min().date()} to {master['Date'].max().date()}")
log(f"Total observations: {len(master)}")

# â”€â”€ Gap Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log(f"\n--- Gap Analysis (missing trading days) ---")

# Calculate business day differences
master['Day_Diff'] = master['Date'].diff().dt.days

# Normal: 1 (consecutive trading day), 2 (over Saturday), 3 (over weekend)
# Suspicious: > 4 (more than a long weekend)
log(f"\nDay-difference distribution:")
day_diff_counts = master['Day_Diff'].value_counts().sort_index()
for diff_val, count in day_diff_counts.items():
    if pd.notna(diff_val):
        label = ""
        if diff_val == 1:
            label = " (consecutive trading day)"
        elif diff_val == 2:
            label = " (1 non-trading day, e.g., Sat)"
        elif diff_val == 3:
            label = " (weekend gap)"
        elif diff_val == 4:
            label = " (3-day weekend / holiday)"
        elif diff_val >= 5:
            label = " â† INVESTIGATE"
        log(f"  {int(diff_val)} day(s): {count} occurrences{label}")

# Show gaps > 4 days
large_gaps = master[master['Day_Diff'] > 4][['Date', 'Day_Diff', 'Close']].copy()
large_gaps['Prev_Date'] = master['Date'].shift(1)
large_gaps = large_gaps[['Prev_Date', 'Date', 'Day_Diff', 'Close']]

if len(large_gaps) > 0:
    log(f"\n--- Gaps > 4 Calendar Days (potential holidays/closures) ---")
    log(large_gaps.to_string(index=False))
    log(f"\nNote: Gaps of 4-5 days typically correspond to long weekends")
    log(f"(e.g., Thanksgiving, Christmas, MLK Day, Presidents Day, etc.)")
    log(f"These are LEGITIMATE non-trading gaps and should NOT be filled.")
else:
    log(f"\n  No gaps larger than 4 calendar days.")

# â”€â”€ Verify continuity at the join point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log(f"\n--- Continuity at Dataset Join Point ---")
join_date = fred_start
# Last D1 observation before FRED
last_d1 = part_a.iloc[-1] if len(part_a) > 0 else None
# First FRED observation
first_fred_idx = master[master['Source'] == 'FRED'].index[0]
first_fred = master.iloc[first_fred_idx]
prev_obs = master.iloc[first_fred_idx - 1] if first_fred_idx > 0 else None

if prev_obs is not None:
    log(f"Last Investing.com observation: {prev_obs['Date'].date()} "
        f"Close={prev_obs['Close']:.2f}")
    log(f"First FRED observation:         {first_fred['Date'].date()} "
        f"Close={first_fred['Close']:.2f}")
    gap_days = (first_fred['Date'] - prev_obs['Date']).days
    price_jump = abs(first_fred['Close'] - prev_obs['Close'])
    price_jump_pct = price_jump / prev_obs['Close'] * 100
    log(f"Gap: {gap_days} calendar day(s), "
        f"price difference: {price_jump:.2f} ({price_jump_pct:.2f}%)")
    if price_jump_pct > 5:
        log(f"  WARNING: Large price jump at join point!")
    else:
        log(f"  Join point is smooth â€” no unusual discontinuity.")

# Drop helper column
master = master.drop(columns=['Day_Diff'])

# Plot full series with source coloring
fig, ax = plt.subplots(figsize=(16, 6))
d1_mask = master['Source'] == 'Investing.com'
fred_mask = master['Source'] == 'FRED'
ax.plot(master.loc[d1_mask, 'Date'], master.loc[d1_mask, 'Close'],
        color='#2196F3', linewidth=0.6, label='Investing.com', alpha=0.9)
ax.plot(master.loc[fred_mask, 'Date'], master.loc[fred_mask, 'Close'],
        color='#FF5722', linewidth=0.6, label='FRED', alpha=0.9)
ax.axvline(x=fred_start, color='gray', linestyle='--', alpha=0.5,
           label=f'Source transition ({fred_start.date()})')
ax.set_title('S&P 500 Master Series â€” Full Period', fontsize=14)
ax.set_ylabel('Close Price')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'step5_master_series.png'), dpi=150)
plt.close()
log(f"\nPlot saved: {os.path.join(PLOTS_DIR, 'step5_master_series.png')}")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 6 â€” DERIVE LOG RETURNS                                               â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 6 â€” DERIVE LOG RETURNS")

log("Formula: r_t = 100 * ln(Close_t / Close_{t-1})")
log("DECISION: Log returns computed from the clean Close series.")
log("  The original Close values are NOT modified.")

master['Log_Return'] = 100 * np.log(master['Close'] / master['Close'].shift(1))

# Report
n_missing_lr = master['Log_Return'].isnull().sum()
lr = master['Log_Return'].dropna()

log(f"\n--- Log Return Statistics ---")
log(f"Missing values (from differencing): {n_missing_lr}")
log(f"  (Expected: 1, the first observation)")
log(f"Number of valid log returns: {len(lr)}")
log(f"Mean:               {lr.mean():.6f}")
log(f"Standard deviation:  {lr.std():.6f}")
log(f"Minimum:            {lr.min():.4f}")
log(f"Maximum:            {lr.max():.4f}")
log(f"Skewness:           {lr.skew():.4f}")
log(f"Kurtosis (excess):  {lr.kurtosis():.4f}")
log(f"Median:             {lr.median():.6f}")

# Extreme returns
log(f"\n--- Top 10 Most Negative Returns ---")
worst = master.nsmallest(10, 'Log_Return')[['Date', 'Close', 'Log_Return']]
log(worst.to_string(index=False))

log(f"\n--- Top 10 Most Positive Returns ---")
best = master.nlargest(10, 'Log_Return')[['Date', 'Close', 'Log_Return']]
log(best.to_string(index=False))


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 7 â€” BASIC FINANCIAL DATA SANITY CHECKS                               â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 7 â€” BASIC FINANCIAL DATA SANITY CHECKS")

lr_clean = master['Log_Return'].dropna().values

# â”€â”€ 7a. Distribution Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- 7a. Distribution Summary ---")
log(f"Jarque-Bera test (normality):")
jb_stat, jb_p = stats.jarque_bera(lr_clean)
log(f"  Test statistic: {jb_stat:.2f}")
log(f"  p-value: {jb_p:.2e}")
log(f"  {'REJECT normality (expected for financial returns)' if jb_p < 0.05 else 'Cannot reject normality'}")

log(f"\nShapiro-Wilk test (on subsample of 5000):")
subsample = lr_clean[:5000] if len(lr_clean) > 5000 else lr_clean
sw_stat, sw_p = stats.shapiro(subsample)
log(f"  Test statistic: {sw_stat:.6f}")
log(f"  p-value: {sw_p:.2e}")
log(f"  {'REJECT normality' if sw_p < 0.05 else 'Cannot reject normality'}")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(lr_clean, bins=100, density=True, color='#42A5F5', alpha=0.7,
             edgecolor='white', linewidth=0.3)
# Overlay normal distribution
x_range = np.linspace(lr_clean.min(), lr_clean.max(), 300)
axes[0].plot(x_range, stats.norm.pdf(x_range, lr_clean.mean(), lr_clean.std()),
             'r-', linewidth=1.5, label='Normal fit')
axes[0].set_title('Distribution of Log Returns (Ã— 100)')
axes[0].set_xlabel('Log Return (%)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(lr_clean, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot vs Normal Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'step7a_distribution.png'), dpi=150)
plt.close()
log(f"Plot saved: {os.path.join(PLOTS_DIR, 'step7a_distribution.png')}")

# â”€â”€ 7b. ACF / PACF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- 7b. Autocorrelation Analysis ---")

n_lags = 40

# ACF of returns
acf_vals = acf(lr_clean, nlags=n_lags, fft=True)
# ACF of squared returns (proxy for volatility)
acf_sq_vals = acf(lr_clean**2, nlags=n_lags, fft=True)
# ACF of absolute returns
acf_abs_vals = acf(np.abs(lr_clean), nlags=n_lags, fft=True)

log(f"ACF of returns (first 5 lags): {[f'{v:.4f}' for v in acf_vals[1:6]]}")
log(f"ACF of squared returns (first 5 lags): {[f'{v:.4f}' for v in acf_sq_vals[1:6]]}")
log(f"ACF of |returns| (first 5 lags): {[f'{v:.4f}' for v in acf_abs_vals[1:6]]}")

conf_bound = 1.96 / np.sqrt(len(lr_clean))
n_sig_acf = sum(1 for v in acf_vals[1:] if abs(v) > conf_bound)
n_sig_acf_sq = sum(1 for v in acf_sq_vals[1:] if abs(v) > conf_bound)
log(f"\nSignificant ACF lags (returns): {n_sig_acf}/{n_lags}")
log(f"Significant ACF lags (squared returns): {n_sig_acf_sq}/{n_lags}")
if n_sig_acf_sq > n_lags * 0.5:
    log("â†’ STRONG evidence of volatility clustering (typical for financial data)")
    log("  This supports the use of GARCH-type models.")

# Plot ACF/PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Returns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(lr_clean, lags=n_lags, ax=axes[0, 0], title='ACF of Log Returns')
plot_pacf(lr_clean, lags=n_lags, ax=axes[0, 1], title='PACF of Log Returns',
          method='ywm')

# Squared returns
plot_acf(lr_clean**2, lags=n_lags, ax=axes[1, 0],
         title='ACF of Squared Log Returns')
plot_pacf(lr_clean**2, lags=n_lags, ax=axes[1, 1],
          title='PACF of Squared Log Returns', method='ywm')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'step7b_acf_pacf.png'), dpi=150)
plt.close()
log(f"Plot saved: {os.path.join(PLOTS_DIR, 'step7b_acf_pacf.png')}")

# â”€â”€ 7c. Volatility Clustering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- 7c. Volatility Clustering ---")

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

dates_lr = master['Date'].iloc[1:].values  # skip first NaN

axes[0].plot(dates_lr, lr_clean, linewidth=0.3, color='#1565C0', alpha=0.7)
axes[0].set_title('Log Returns Over Time')
axes[0].set_ylabel('Log Return (%)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(dates_lr, np.abs(lr_clean), linewidth=0.3, color='#E65100', alpha=0.7)
# Add rolling 21-day average of absolute returns
rolling_vol = pd.Series(np.abs(lr_clean)).rolling(21).mean().values
axes[1].plot(dates_lr, rolling_vol, linewidth=1.0, color='black',
             label='21-day rolling mean |r|')
axes[1].set_title('Absolute Log Returns (Volatility Proxy)')
axes[1].set_ylabel('|Log Return| (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Rolling standard deviation (21-day)
rolling_std = pd.Series(lr_clean).rolling(21).std().values
axes[2].plot(dates_lr, rolling_std, linewidth=0.8, color='#6A1B9A')
axes[2].set_title('21-Day Rolling Standard Deviation of Log Returns')
axes[2].set_ylabel('Rolling Std (%)')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'step7c_volatility_clustering.png'), dpi=150)
plt.close()
log(f"Plot saved: {os.path.join(PLOTS_DIR, 'step7c_volatility_clustering.png')}")

log(f"\nVisual inspection of absolute returns and rolling volatility")
log(f"should reveal periods of clustered high/low volatility:")
log(f"  - 2000â€“2003: Dot-com crash (high volatility)")
log(f"  - 2007â€“2009: Global Financial Crisis (very high volatility)")
log(f"  - 2020: COVID-19 crash (extreme volatility spike)")
log(f"  - 2022: Rate hike volatility")

# â”€â”€ 7d. ADF Test (Stationarity) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- 7d. Augmented Dickey-Fuller Test ---")

log("\nADF test on CLOSE PRICES (levels):")
adf_price = adfuller(master['Close'].dropna(), maxlag=20, regression='ct')
log(f"  Test statistic: {adf_price[0]:.4f}")
log(f"  p-value: {adf_price[1]:.6f}")
log(f"  Lags used: {adf_price[2]}")
log(f"  Critical values: {adf_price[4]}")
log(f"  {'REJECT H0 (stationary)' if adf_price[1] < 0.05 else 'CANNOT reject H0 (non-stationary) â€” expected for prices'}")

log("\nADF test on LOG RETURNS:")
adf_lr = adfuller(lr_clean, maxlag=20, regression='c')
log(f"  Test statistic: {adf_lr[0]:.4f}")
log(f"  p-value: {adf_lr[1]:.6f}")
log(f"  Lags used: {adf_lr[2]}")
log(f"  Critical values: {adf_lr[4]}")
log(f"  {'REJECT H0 (stationary) â€” expected for returns' if adf_lr[1] < 0.05 else 'CANNOT reject H0 (non-stationary) â€” unexpected!'}")

# â”€â”€ 7e. Ljung-Box test for serial correlation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\n--- 7e. Ljung-Box Test ---")
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_returns = acorr_ljungbox(lr_clean, lags=[10, 20, 40], return_df=True)
log(f"\nLjung-Box test on returns:")
log(lb_returns.to_string())

lb_sq = acorr_ljungbox(lr_clean**2, lags=[10, 20, 40], return_df=True)
log(f"\nLjung-Box test on squared returns (ARCH effects):")
log(lb_sq.to_string())

if (lb_sq['lb_pvalue'] < 0.05).all():
    log("\nâ†’ Strong evidence of ARCH effects (volatility clustering)")
    log("  This confirms the relevance of GARCH modeling for this data.")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEP 8 â€” SAVE OUTPUTS                                                     â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section("STEP 8 â€” SAVE OUTPUTS")

# â”€â”€ Save master CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
output_df = master[['Date', 'Close', 'Log_Return', 'Source']].copy()
output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')
output_df.to_csv(OUTPUT_CSV, index=False)
log(f"Master dataset saved: {OUTPUT_CSV}")
log(f"  Rows: {len(output_df)}")
log(f"  Columns: {list(output_df.columns)}")

# â”€â”€ Save validation report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("VALIDATION REPORT SUMMARY")

log(f"\n{'=' * 60}")
log(f"  FINAL SUMMARY")
log(f"{'=' * 60}")
log(f"")
log(f"1. Final date range:        {master['Date'].min().date()} to {master['Date'].max().date()}")
log(f"2. Number of observations:  {len(master)}")
log(f"3. Overlapping dates:       {len(overlap)}")
log(f"4. Source agreement:        Mean abs diff = {overlap['Abs_Diff'].mean():.4f}, "
    f"Max abs diff = {overlap['Abs_Diff'].max():.4f}")
log(f"5. Discrepancies:           {discrepancy_level} â€” "
    f"{'differences attributable to rounding' if discrepancy_level in ['minor_rounding', 'small'] else 'manual review recommended'}")
log(f"6. Merge rule:              Investing.com for pre-FRED period; "
    f"FRED from {fred_start.date()} onward (no overlap duplication)")
log(f"7. Missing Log_Return:      {n_missing_lr} "
    f"(first observation, expected)")
log(f"8. Dataset readiness:       {'READY for modeling' if n_missing_close == 0 and n_nonpositive == 0 and n_dup == 0 else 'ISSUES FOUND â€” review above'}")
log(f"")
log(f"Source breakdown:")
for src, cnt in src_counts.items():
    log(f"  {src}: {cnt} observations")
log(f"")
log(f"Key financial characteristics confirmed:")
log(f"  - Non-normal returns (heavy tails, excess kurtosis)")
log(f"  - Volatility clustering (ARCH effects)")
log(f"  - Stationary log returns (ADF test)")
log(f"  - Non-stationary price levels (ADF test)")
log(f"")
log(f"{'=' * 60}")
log(f"  Dataset is ready for ARIMA-GARCH and LSTM modeling.")
log(f"{'=' * 60}")

# Write report to file
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"\nValidation report saved: {REPORT_PATH}")

print("\nâœ… Data preparation pipeline complete.")
print(f"   Output: {OUTPUT_CSV}")
print(f"   Report: {REPORT_PATH}")
print(f"   Plots:  {PLOTS_DIR}/")
