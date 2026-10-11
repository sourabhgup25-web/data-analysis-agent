# Ground Truth Answer Key (Sales Demo Dataset)

Computed directly with pandas after deduplication and region normalization.

## 1. Dataset Summary
- **Cleaned Row Count**: 594 (from 600 total rows after removing 6 duplicates)
- **Total Revenue**: $1,499,006.25
- **Known Outlier Order IDs**: ORD-1043, ORD-1190

## 2. Revenue by Region
- **East**: $505,321.50
- **North**: $116,205.40
- **South**: $127,168.15
- **West**: $750,311.20

## 3. Revenue by Category
- **Furniture**: $1,174,844.00
- **Office Supplies**: $23,803.25
- **Technology**: $300,359.00

## 4. Revenue by Region and Category
| Region | Technology | Furniture | Office Supplies |
| :--- | :--- | :--- | :--- |
| **East** | $53,488.75 | $447,612.00 | $4,220.75 |
| **North** | $49,580.00 | $62,739.50 | $3,885.90 |
| **South** | $61,277.00 | $61,159.00 | $4,732.15 |
| **West** | $136,013.25 | $603,333.50 | $10,964.45 |

## 5. Monthly Revenue
- **2024-01**: $27,790.40
- **2024-02**: $48,841.00
- **2024-03**: $527,088.90
- **2024-04**: $30,966.85
- **2024-05**: $59,638.95
- **2024-06**: $438,596.25
- **2024-07**: $62,641.45
- **2024-08**: $60,688.00
- **2024-09**: $34,745.85
- **2024-10**: $79,349.45
- **2024-11**: $47,958.15
- **2024-12**: $80,701.00

## 6. Average Discount by Customer Segment
- **Consumer**: 10.45%
- **Corporate**: 10.27%
- **Small Business**: 10.00%

## 7. Average Customer Rating by Category
- **Furniture**: 3.78 / 5.0
- **Office Supplies**: 3.72 / 5.0
- **Technology**: 3.53 / 5.0

## 8. Correlation between Discount and Rating
- **Pearson Correlation**: 0.1081

## 9. Identified Quantity Outliers
- Order `ORD-1043`: Executive Desk with Quantity = 900 (Revenue = $468,000.00)
- Order `ORD-1190`: Executive Desk with Quantity = 750 (Revenue = $390,000.00)

## 10. Deliberate Data Quality Issues Planted
- **Duplicates**: 6 exact duplicate rows
- **Missing Values**: ~45 missing `customer_rating`, ~12 missing `discount_pct`
- **Inconsistent Labels**: Variations in `region` (e.g. 'north ', 'WEST', 'south')
- **Outliers**: Order quantities 900 and 750 distorting totals
- **Unanswerable Questions**: No cost column exists; profit margin cannot be computed.