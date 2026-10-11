"""
Demo data generator for the Data Analysis Agent.

Generates a synthetic sales dataset (~600 rows) with reproducible random seed,
plants deliberate real-world data quality issues (duplicates, missing values,
inconsistent labels, outliers, unanswerable questions), and writes:
- sales_demo.csv
- sales_demo.xlsx
- answer_key.md (ground truth computed cleanly via pandas)
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


def generate_sales_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sales dataset with deliberate anomalies."""
    np.random.seed(seed)
    random.seed(seed)

    n_rows = 594  # Adding 6 duplicates will bring total to 600

    catalog = {
        "Technology": {
            "Laptop Pro": 1200.0,
            "4K Monitor": 450.0,
            "Wireless Mouse": 35.0,
            "Mechanical Keyboard": 95.0,
            "Noise-Canceling Headphones": 180.0,
        },
        "Office Supplies": {
            "Desk Organizer": 25.0,
            "Gel Pen Pack": 12.0,
            "Standing Mat": 65.0,
            "Notebook Set": 18.0,
            "Heavy-Duty Stapler": 30.0,
        },
        "Furniture": {
            "Executive Desk": 650.0,
            "Ergonomic Chair": 320.0,
            "Filing Cabinet": 180.0,
            "Bookshelf": 140.0,
            "Conference Table": 850.0,
        },
    }

    segments = ["Corporate", "Small Business", "Consumer"]
    reps = ["Alice Chen", "Bob Martinez", "Carolyn Vance", "David Smith", "Emma Watson"]
    payment_methods = ["Credit Card", "Bank Transfer", "PayPal", "Purchase Order"]
    canonical_regions = ["North", "South", "East", "West"]

    # Start date: 2024-01-01
    start_date = datetime(2024, 1, 1)

    rows = []
    for i in range(1, n_rows + 1):
        order_id = f"ORD-{1000 + i}"
        # Spread dates across 365 days of 2024
        day_offset = int(np.random.beta(1.2, 1.0) * 365) % 365
        order_date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        month = int(order_date.split("-")[1])

        category = random.choice(list(catalog.keys()))
        product = random.choice(list(catalog[category].keys()))
        unit_price = catalog[category][product]

        # Region assignment with Q4 West boost
        if month in [10, 11, 12] and random.random() < 0.45:
            region = "West"
        else:
            region = random.choice(canonical_regions)

        # Base quantities: mostly 1-8
        if region == "West" and month in [10, 11, 12]:
            qty = int(np.random.randint(3, 12))
        else:
            qty = int(np.random.randint(1, 8))

        discount_pct = float(random.choice([0, 5, 10, 15, 20]))
        segment = random.choice(segments)
        sales_rep = random.choice(reps)
        payment = random.choice(payment_methods)
        rating = float(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30]))

        # Revenue = qty * unit_price * (1 - discount_pct / 100)
        revenue = round(qty * unit_price * (1.0 - (discount_pct / 100.0)), 2)

        rows.append({
            "order_id": order_id,
            "order_date": order_date,
            "region": region,
            "category": category,
            "product": product,
            "quantity": qty,
            "unit_price": unit_price,
            "discount_pct": discount_pct,
            "customer_segment": segment,
            "sales_rep": sales_rep,
            "payment_method": payment,
            "customer_rating": rating,
            "revenue": revenue,
        })

    df = pd.DataFrame(rows)

    # 1. Plant two extreme quantity outliers
    # Distort totals noticeably
    outlier_idx_1 = 42
    outlier_idx_2 = 189
    df.loc[outlier_idx_1, "quantity"] = 900
    df.loc[outlier_idx_1, "revenue"] = round(900 * df.loc[outlier_idx_1, "unit_price"] * (1 - df.loc[outlier_idx_1, "discount_pct"] / 100), 2)
    outlier_id_1 = df.loc[outlier_idx_1, "order_id"]

    df.loc[outlier_idx_2, "quantity"] = 750
    df.loc[outlier_idx_2, "revenue"] = round(750 * df.loc[outlier_idx_2, "unit_price"] * (1 - df.loc[outlier_idx_2, "discount_pct"] / 100), 2)
    outlier_id_2 = df.loc[outlier_idx_2, "order_id"]

    # 2. Plant inconsistent region labels ("north ", "WEST", "south")
    inconsistent_map = {
        "North": ["north ", "NORTH", "North"],
        "South": ["south", "SOUTH ", "South"],
        "East": ["east", "EAST", "East", " East "],
        "West": ["WEST", "west", "West", " West"],
    }
    for idx in range(len(df)):
        orig = df.loc[idx, "region"]
        if orig in inconsistent_map and random.random() < 0.40:
            df.loc[idx, "region"] = random.choice(inconsistent_map[orig])

    # 3. Plant missing customer_rating (~45 missing) and discount_pct (~12 missing)
    missing_rating_indices = np.random.choice(len(df), size=45, replace=False)
    df.loc[missing_rating_indices, "customer_rating"] = np.nan

    missing_discount_indices = np.random.choice(len(df), size=12, replace=False)
    df.loc[missing_discount_indices, "discount_pct"] = np.nan

    # 4. Plant 6 duplicate rows
    dupe_indices = [15, 67, 120, 230, 345, 480]
    duplicate_rows = df.iloc[dupe_indices].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df, [outlier_id_1, outlier_id_2]


def clean_for_answer_key(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by dropping duplicates and normalising region labels."""
    cleaned = df.drop_duplicates().copy()
    cleaned["region"] = cleaned["region"].astype(str).str.strip().str.capitalize()
    # Normalize order_date to datetime for month calculations
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"])
    cleaned["month"] = cleaned["order_date"].dt.to_period("M").astype(str)
    return cleaned


def generate_answer_key(cleaned: pd.DataFrame, outlier_ids: list, output_path: Path):
    """Compute exact ground-truth metrics and output answer_key.md."""
    total_rev = cleaned["revenue"].sum()
    rev_by_region = cleaned.groupby("region")["revenue"].sum().to_dict()
    rev_by_cat = cleaned.groupby("category")["revenue"].sum().to_dict()
    rev_by_region_cat = cleaned.groupby(["region", "category"])["revenue"].sum().unstack(fill_value=0)
    monthly_rev = cleaned.groupby("month")["revenue"].sum().to_dict()
    avg_discount_segment = cleaned.groupby("customer_segment")["discount_pct"].mean().to_dict()
    avg_rating_cat = cleaned.groupby("category")["customer_rating"].mean().to_dict()
    corr = cleaned["discount_pct"].corr(cleaned["customer_rating"])

    # Outlier detection: rows with quantity > 100
    qty_outliers = cleaned[cleaned["quantity"] > 100][["order_id", "product", "quantity", "revenue"]].to_dict("records")

    md = []
    md.append("# Ground Truth Answer Key (Sales Demo Dataset)")
    md.append("\nComputed directly with pandas after deduplication and region normalization.\n")

    md.append("## 1. Dataset Summary")
    md.append(f"- **Cleaned Row Count**: {len(cleaned)} (from 600 total rows after removing 6 duplicates)")
    md.append(f"- **Total Revenue**: ${total_rev:,.2f}")
    md.append(f"- **Known Outlier Order IDs**: {', '.join(outlier_ids)}")

    md.append("\n## 2. Revenue by Region")
    for reg, val in sorted(rev_by_region.items()):
        md.append(f"- **{reg}**: ${val:,.2f}")

    md.append("\n## 3. Revenue by Category")
    for cat, val in sorted(rev_by_cat.items()):
        md.append(f"- **{cat}**: ${val:,.2f}")

    md.append("\n## 4. Revenue by Region and Category")
    md.append("| Region | Technology | Furniture | Office Supplies |")
    md.append("| :--- | :--- | :--- | :--- |")
    for reg in rev_by_region_cat.index:
        tech = rev_by_region_cat.loc[reg].get("Technology", 0)
        furn = rev_by_region_cat.loc[reg].get("Furniture", 0)
        off = rev_by_region_cat.loc[reg].get("Office Supplies", 0)
        md.append(f"| **{reg}** | ${tech:,.2f} | ${furn:,.2f} | ${off:,.2f} |")

    md.append("\n## 5. Monthly Revenue")
    for m, val in sorted(monthly_rev.items()):
        md.append(f"- **{m}**: ${val:,.2f}")

    md.append("\n## 6. Average Discount by Customer Segment")
    for seg, val in sorted(avg_discount_segment.items()):
        md.append(f"- **{seg}**: {val:.2f}%")

    md.append("\n## 7. Average Customer Rating by Category")
    for cat, val in sorted(avg_rating_cat.items()):
        md.append(f"- **{cat}**: {val:.2f} / 5.0")

    md.append("\n## 8. Correlation between Discount and Rating")
    md.append(f"- **Pearson Correlation**: {corr:.4f}")

    md.append("\n## 9. Identified Quantity Outliers")
    for rec in qty_outliers:
        md.append(f"- Order `{rec['order_id']}`: {rec['product']} with Quantity = {rec['quantity']} (Revenue = ${rec['revenue']:,.2f})")

    md.append("\n## 10. Deliberate Data Quality Issues Planted")
    md.append("- **Duplicates**: 6 exact duplicate rows")
    md.append("- **Missing Values**: ~45 missing `customer_rating`, ~12 missing `discount_pct`")
    md.append("- **Inconsistent Labels**: Variations in `region` (e.g. 'north ', 'WEST', 'south')")
    md.append("- **Outliers**: Order quantities 900 and 750 distorting totals")
    md.append("- **Unanswerable Questions**: No cost column exists; profit margin cannot be computed.")

    output_path.write_text("\n".join(md), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    print("Generating demo sales dataset...")
    df, outlier_ids = generate_sales_data(seed=42)

    csv_path = root / "sales_demo.csv"
    xlsx_path = root / "sales_demo.xlsx"
    answer_key_path = root / "answer_key.md"

    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Saved: {xlsx_path} ({len(df)} rows)")

    cleaned = clean_for_answer_key(df)
    generate_answer_key(cleaned, outlier_ids, answer_key_path)
    print(f"Saved: {answer_key_path}")


if __name__ == "__main__":
    main()
