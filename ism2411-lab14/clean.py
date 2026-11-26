"""
Self-Assessment Reflection
1. Most ambiguous decision (drop, fill, or flag):
   Filling missing numeric values with the column mean is the most ambiguous call
   because it assumes missing rows look like an "average" order. Before doing this
   on real company data, I would want to know whether missing values came from a
   specific product category, store region, or POS system glitch—and check whether
   the distribution is skewed enough that the median would be safer than the mean.

2. Why each decision was made (comment check):
   All comments below explain both what the code does and the business reason behind
   it (e.g., preserving negative returns for accurate net revenue vs. dropping 9,999
   placeholder quantities that would distort inventory and sales totals).

3. Most unexpected finding from .describe():
   The biggest surprise in .describe() was how dramatically a small number of 9,999 
   placeholder entries inflated the quantity_sold and revenue averages compared to 
   their medians. Seeing the mean sit far above the 50th percentile (median) showed 
   how easily a handful of bad data-entry rows can distort company-wide sales metrics 
   if you rely on averages without checking distribution tails first.
"""

import pandas as pd

# --- EXERCISE 1 ---

df = pd.read_csv("data/messy_sales.csv")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nMissing value percentages:")
print((df.isnull().mean() * 100).round(2))

# Data quality issues observed from .info() and .isnull().sum():
# 1. unit_price is stored as dtype 'object' instead of float because values include '$' and ',' characters.
# 2. order_date is stored as dtype 'object' instead of datetime64 and contains mixed date formats / invalid text.
# 3. Multiple columns have missing values: customer_id has 7.54% missing (362 rows), region has 2.31% missing (111),
#    quantity_sold has 1.96% missing (94), order_id has 0.65% missing (31), and order_date has 0.52% missing (25).

# --- EXERCISE 2 ---

df.columns = df.columns.str.lower().str.replace(" ", "_")
print(df.columns.tolist())

# --- EXERCISE 3 ---

print("\nShape before handling missing values:", df.shape)

# Drop rows missing order_id or order_date because a transaction without an ID
# or timestamp cannot be audited or grouped into time-based sales reports.
df = df.dropna(subset=["order_id", "order_date"])

# Fill numeric columns that have a small number of missing values using the column mean
# so we preserve valid transaction rows without skewing the dataset's average behavior.
# (Note: unit_price is still text due to '$' symbols and is cleaned/filled in Exercise 4.)
numeric_cols = df.select_dtypes(include=["number"]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Fill any remaining missing text/categorical labels with "Unknown" rather than dropping valid sales.
categorical_cols = ["product_category", "region", "customer_id"]
for col in categorical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("Unknown")

print("Shape after handling missing values:", df.shape)
print("Remaining missing values:\n", df.isnull().sum())

# --- EXERCISE 4 ---
# Convert order_date to datetime; mixed formats or junk text like 'TBD' become NaT.
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
nat_count = df["order_date"].isnull().sum()
print(f"\nNumber of dates that became NaT: {nat_count}")

# Drop rows where order_date could not be parsed so time-series analysis in Lab 15 doesn't break.
df = df.dropna(subset=["order_date"])

# Strip '$' and ',' from unit_price before casting to float (preserves negative signs on returns),
# then fill any missing prices with the column mean so all line items can calculate revenue.
df["unit_price"] = (
    df["unit_price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["unit_price"] = df["unit_price"].fillna(df["unit_price"].mean()).astype(float)

# Clean and cast quantity_sold to integer (filling any newly uncovered nulls with the rounded mean).
df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce")
df["quantity_sold"] = df["quantity_sold"].fillna(round(df["quantity_sold"].mean())).astype(int)

# Calculate line-item revenue rounded to 2 decimal places for Module 15 charts.
df["revenue"] = (df["unit_price"] * df["quantity_sold"]).round(2)

print("\nUpdated dtypes:")
print(df.dtypes)

# --- EXERCISE 5 ---

dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")

# Remove exact duplicate records so the same order is not double-counted in revenue totals.
if dup_count > 0:
    df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# --- EXERCISE 6 ---

print(df.describe())

# Suspicious statistics & business explanations:
# 1. Negative minimum values in unit_price / revenue: These represent customer returns
#    and refunds—real money returned to customers—so we keep them for accurate net sales.
# 2. Maximum quantity_sold of 9,999 (pulling the mean far above the median): A single retail
#    line item of 9,999 units is a classic manual placeholder/test entry rather than a real order.

# Filter out the 9,999 placeholder rows because they artificially inflate total units and revenue,
# while retaining negative revenue rows because returns are legitimate business transactions.
df = df[df["quantity_sold"] != 9999]

print("\nShape after removing quantity_sold == 9999 placeholders:", df.shape)

# --- EXERCISE 7 ---

df.to_csv("data/clean_sales.csv", index=False)

# Reload without parse_dates (order_date comes back as object/text because CSVs don't store date types)
check_raw = pd.read_csv("data/clean_sales.csv")
print("\nReloaded CSV without parse_dates:")
check_raw.info()

# Reload with parse_dates=["order_date"] (confirms order_date parses cleanly as datetime64)
check_parsed = pd.read_csv("data/clean_sales.csv", parse_dates=["order_date"])
print("\nReloaded CSV with parse_dates=['order_date']:")
check_parsed.info()
print("\nFinal clean dataset shape:", check_parsed.shape)