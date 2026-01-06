# SPEC: KAX — Morphogen Analytical eXpressions

> 🔬 **Implementation Status: RESEARCH SPECIFICATION**
> This is a proposed feature for future Morphogen versions. Not yet implemented.

**Version:** 1.0
**Status:** Proposed (post-v1.0)
**Last Updated:** 2025-11-15
**Related:** bi-domain.md, ../adr/007-gpu-first-domains.md

---

## Overview

**KAX (Morphogen Analytical eXpressions)** is a declarative expression language for business intelligence and analytical computations in Morphogen. It is inspired by **DAX (Data Analysis Expressions)** from Microsoft Power BI and SSAS Tabular, but designed to compile into **GPU-native operator graphs**.

### Why KAX?

KAX brings semantic modeling to Morphogen's BIDomain:

1. **Measures & Calculated Columns** — Define reusable analytical computations
2. **Filter Context** — DAX-style context manipulation (`CALCULATE`, `FILTER`, `ALL`)
3. **Time Intelligence** — Date-aware functions (`SAMEPERIODLASTYEAR`, `DATEADD`, etc.)
4. **Row Context vs Filter Context** — Semantic distinction like DAX
5. **GPU Compilation** — Expressions lower to GPU operators for maximum performance
6. **Cross-Domain Integration** — KAX expressions can drive simulations, visualizations, ML pipelines

---

## Design Principles

### 1. Declarative

KAX expressions describe **what** to compute, not **how**.

The engine decides:
- GPU operator selection
- Compression strategies
- Batching and scheduling
- Memory management

Example:

```morphogen
Sales[TotalRevenue] := SUM(Sales[Amount])
```

Compiles to:

```
gpu.dict_encode(Sales[ProductID])
→ gpu.segmented_groupby(agg: Sum)
→ result
```

---

### 2. Columnar & GPU-Aware

Unlike row-by-row interpreted languages, KAX compiles into **columnar GPU operator DAGs**.

Example:

```morphogen
Sales[Margin] := DIVIDE(Sales[Profit], Sales[Revenue])
```

Compiles to:

```
gpu.column_divide(
    numerator: Sales[Profit],
    denominator: Sales[Revenue]
)
```

No row-level iteration — pure vectorized GPU execution.

---

### 3. Domain-Composable

KAX expressions integrate with all Morphogen domains.

**BI → Visualization:**

```morphogen
let total_sales = Sales[TotalRevenue]
viz.bar_chart(total_sales)
```

**BI → Simulation:**

```morphogen
let avg_temp = Sensors[AvgTemperature]
physics.thermal_ode(initial_temp: avg_temp)
```

**BI → ML:**

```morphogen
let customer_ltv = Customers[LifetimeValue]
ml.train_gpu(features: [customer_ltv, ...], target: Customers[Churn])
```

---

### 4. Measures vs Calculated Columns

**Calculated Columns** — Computed once per row, stored

```morphogen
Orders[DayOfWeek] := WEEKDAY(Orders[Date])
```

**Measures** — Computed dynamically based on filter context

```morphogen
Sales[TotalRevenue] := SUM(Sales[Amount])
```

KAX distinguishes these semantically and compiles them differently.

---

## Syntax

### Assignment

```morphogen
TableName[ColumnName] := Expression
```

Examples:

```morphogen
Sales[TotalRevenue] := SUM(Sales[Amount])
Sales[MarginPct] := DIVIDE(Sales[Profit], Sales[Revenue])
Orders[Year] := YEAR(Orders[Date])
```

---

### Aggregation Functions

```morphogen
SUM(column)
COUNT(column)
AVERAGE(column)
MIN(column)
MAX(column)
STDEV(column)
VARIANCE(column)
```

Examples:

```morphogen
Sales[TotalSales] := SUM(Sales[Amount])
Sales[AvgOrderValue] := AVERAGE(Sales[Amount])
Sales[MaxDiscount] := MAX(Sales[Discount])
Sales[OrderCount] := COUNT(Sales[OrderID])
```

GPU compilation:

```morphogen
SUM(Sales[Amount])
→ gpu.agg_sum(Sales[Amount])
```

---

### Arithmetic & Comparison

```morphogen
+  -  *  /  %          // Arithmetic
=  <>  <  >  <=  >=    // Comparison
&&  ||  !              // Logical
```

Examples:

```morphogen
Sales[Margin] := Sales[Revenue] - Sales[Cost]
Sales[MarginPct] := DIVIDE(Sales[Margin], Sales[Revenue]) * 100
Sales[IsHighValue] := Sales[Amount] > 1000
```

GPU compilation:

```morphogen
Sales[Revenue] - Sales[Cost]
→ gpu.column_subtract(Sales[Revenue], Sales[Cost])
```

---

### Filter Context Manipulation

#### CALCULATE

`CALCULATE` modifies the filter context for a measure.

Syntax:

```morphogen
CALCULATE(expression, filter1, filter2, ...)
```

Examples:

```morphogen
// Total sales for 2024 only
Sales[2024Revenue] := CALCULATE(
    SUM(Sales[Amount]),
    Year(Sales[Date]) = 2024
)

// Sales in USA
Sales[USASales] := CALCULATE(
    SUM(Sales[Amount]),
    Sales[Country] = "USA"
)

// High-value orders
Sales[HighValueSales] := CALCULATE(
    SUM(Sales[Amount]),
    Sales[Amount] > 1000
)
```

GPU compilation:

```morphogen
CALCULATE(SUM(Sales[Amount]), Year = 2024)
→ gpu.dict_encode(Calendar[Year])
→ gpu.predicate(Year, EQ, 2024)
→ gpu.bitmap_filter(Sales, bitmap)
→ gpu.agg_sum(Sales[Amount])
```

---

#### FILTER

`FILTER` returns a filtered table.

Syntax:

```morphogen
FILTER(table, condition)
```

Examples:

```morphogen
// All high-value customers
HighValueCustomers := FILTER(
    Customers,
    Customers[LifetimeValue] > 10000
)

// Recent orders
RecentOrders := FILTER(
    Orders,
    Orders[Date] > DATE(2024, 1, 1)
)
```

GPU compilation:

```morphogen
FILTER(Customers, Customers[LifetimeValue] > 10000)
→ gpu.predicate(Customers[LifetimeValue], GT, 10000)
→ gpu.bitmap_filter(Customers, bitmap)
```

---

#### ALL

`ALL` removes filters from a table or column.

Syntax:

```morphogen
ALL(table)
ALL(column)
```

Examples:

```morphogen
// Total sales across all products (ignore product filter)
Sales[TotalAllProducts] := CALCULATE(
    SUM(Sales[Amount]),
    ALL(Products)
)

// Percent of total
Sales[PctOfTotal] := DIVIDE(
    SUM(Sales[Amount]),
    CALCULATE(SUM(Sales[Amount]), ALL(Sales))
)
```

---

#### ALLEXCEPT

`ALLEXCEPT` removes all filters except specified columns.

Syntax:

```morphogen
ALLEXCEPT(table, column1, column2, ...)
```

Example:

```morphogen
// Total sales per region, ignoring all other filters
Sales[TotalByRegion] := CALCULATE(
    SUM(Sales[Amount]),
    ALLEXCEPT(Sales, Sales[Region])
)
```

---

### Iterators

Iterators operate row-by-row within filter context.

#### SUMX

`SUMX` iterates over a table and sums an expression.

Syntax:

```morphogen
SUMX(table, expression)
```

Examples:

```morphogen
// Total revenue (price × quantity per row)
Sales[TotalRevenue] := SUMX(
    Sales,
    Sales[Price] * Sales[Quantity]
)

// Weighted average
Sales[WeightedAvg] := DIVIDE(
    SUMX(Sales, Sales[Value] * Sales[Weight]),
    SUM(Sales[Weight])
)
```

GPU compilation:

```morphogen
SUMX(Sales, Sales[Price] * Sales[Quantity])
→ gpu.column_multiply(Sales[Price], Sales[Quantity])
→ gpu.agg_sum(result)
```

---

#### AVERAGEX

`AVERAGEX` iterates and computes average.

Syntax:

```morphogen
AVERAGEX(table, expression)
```

Example:

```morphogen
Sales[AvgMarginPct] := AVERAGEX(
    Sales,
    DIVIDE(Sales[Profit], Sales[Revenue])
)
```

---

#### COUNTX

`COUNTX` counts rows where expression is not blank.

Syntax:

```morphogen
COUNTX(table, expression)
```

Example:

```morphogen
Sales[OrdersWithDiscount] := COUNTX(
    Sales,
    Sales[Discount]
)
```

---

#### FILTERX (Custom)

Morphogen extension: `FILTERX` for GPU-optimized filtering with expressions.

Syntax:

```morphogen
FILTERX(table, expression)
```

Example:

```morphogen
HighMarginOrders := FILTERX(
    Orders,
    DIVIDE(Orders[Profit], Orders[Revenue]) > 0.3
)
```

---

### Time Intelligence

#### Date Functions

```morphogen
YEAR(date)
MONTH(date)
DAY(date)
QUARTER(date)
WEEKDAY(date)
DATEADD(date, number, interval)
DATEDIFF(date1, date2, interval)
```

Examples:

```morphogen
Orders[Year] := YEAR(Orders[Date])
Orders[Quarter] := QUARTER(Orders[Date])
Orders[DaysSinceOrder] := DATEDIFF(Orders[Date], TODAY(), DAY)
```

---

#### Time Intelligence Functions

```morphogen
SAMEPERIODLASTYEAR(dates)
PREVIOUSMONTH(dates)
PREVIOUSQUARTER(dates)
PREVIOUSYEAR(dates)
DATEADD(dates, number, interval)
```

Examples:

```morphogen
// Year-over-year growth
Sales[YoY] :=
    Sales[TotalRevenue] -
    CALCULATE(
        Sales[TotalRevenue],
        SAMEPERIODLASTYEAR(Calendar[Date])
    )

// Month-over-month
Sales[MoM] :=
    Sales[TotalRevenue] -
    CALCULATE(
        Sales[TotalRevenue],
        PREVIOUSMONTH(Calendar[Date])
    )

// Rolling 12 months
Sales[Rolling12M] := CALCULATE(
    SUM(Sales[Amount]),
    DATESINPERIOD(Calendar[Date], LASTDATE(Calendar[Date]), -12, MONTH)
)
```

GPU compilation:

```morphogen
SAMEPERIODLASTYEAR(Calendar[Date])
→ gpu.date_shift(Calendar[Date], -1, YEAR)
→ gpu.predicate(Calendar[Date], IN, shifted_dates)
→ gpu.bitmap_filter
```

---

### Logical Functions

```morphogen
IF(condition, true_value, false_value)
AND(condition1, condition2, ...)
OR(condition1, condition2, ...)
NOT(condition)
SWITCH(expression, value1, result1, value2, result2, ..., default)
```

Examples:

```morphogen
Sales[Category] := IF(
    Sales[Amount] > 1000,
    "High Value",
    "Standard"
)

Sales[Segment] := SWITCH(
    Sales[Region],
    "North", "Domestic",
    "South", "Domestic",
    "East", "International",
    "West", "International",
    "Unknown"
)
```

---

### Text Functions

```morphogen
CONCATENATE(text1, text2)
LEFT(text, num_chars)
RIGHT(text, num_chars)
MID(text, start, num_chars)
UPPER(text)
LOWER(text)
LEN(text)
FIND(find_text, within_text)
SUBSTITUTE(text, old_text, new_text)
```

Examples:

```morphogen
Customers[FullName] := CONCATENATE(
    Customers[FirstName],
    " ",
    Customers[LastName]
)

Products[InitialLetter] := LEFT(Products[Name], 1)
```

---

### Math Functions

```morphogen
ABS(number)
ROUND(number, decimals)
FLOOR(number)
CEILING(number)
SQRT(number)
POWER(number, power)
EXP(number)
LN(number)
LOG(number, base)
MOD(number, divisor)
```

Examples:

```morphogen
Sales[RoundedRevenue] := ROUND(Sales[Revenue], 2)
Sales[MarginAbs] := ABS(Sales[Margin])
```

---

### Statistical Functions

```morphogen
STDEV.S(column)     // Sample standard deviation
STDEV.P(column)     // Population standard deviation
VAR.S(column)       // Sample variance
VAR.P(column)       // Population variance
MEDIAN(column)
PERCENTILE(column, k)
RANK(value, column, order)
```

Examples:

```morphogen
Sales[StdDevAmount] := STDEV.S(Sales[Amount])
Sales[MedianRevenue] := MEDIAN(Sales[Revenue])
Sales[P95Amount] := PERCENTILE(Sales[Amount], 0.95)
```

---

## Row Context vs Filter Context

### Row Context

- Created by: **Calculated columns**, **iterators** (`SUMX`, `AVERAGEX`, etc.)
- Scope: **Single row**
- Aggregations: **Not allowed** (use iterators instead)

Example:

```morphogen
// Row context — computed for each row
Orders[TotalPrice] := Orders[Quantity] * Orders[UnitPrice]
```

---

### Filter Context

- Created by: **Measures**, **CALCULATE**, **filters**
- Scope: **Set of rows** (filtered table)
- Aggregations: **Required** (e.g., `SUM`, `COUNT`)

Example:

```morphogen
// Filter context — aggregates over current filter
Sales[TotalRevenue] := SUM(Sales[Amount])
```

---

### Context Transition

**Iterators transition from filter context to row context.**

Example:

```morphogen
// Filter context
Sales[AvgMarginPct] := AVERAGEX(
    Sales,                              // Filter context
    DIVIDE(Sales[Profit], Sales[Revenue]) // Row context (per row)
)
```

GPU compilation handles this efficiently:

```
gpu.column_divide(Sales[Profit], Sales[Revenue])  // Row context → vectorized
→ gpu.agg_avg(result)                             // Back to filter context
```

---

## GPU Compilation

KAX expressions compile into GPU operator DAGs.

### Example 1: Simple Aggregation

**KAX:**

```morphogen
Sales[TotalRevenue] := SUM(Sales[Amount])
```

**GPU DAG:**

```
gpu.agg_sum(Sales[Amount])
→ result
```

---

### Example 2: Filtered Measure

**KAX:**

```morphogen
Sales[2024Revenue] := CALCULATE(
    SUM(Sales[Amount]),
    YEAR(Sales[Date]) = 2024
)
```

**GPU DAG:**

```
gpu.extract_year(Sales[Date])
→ gpu.predicate(year_col, EQ, 2024)
→ gpu.bitmap_filter(Sales, bitmap)
→ gpu.agg_sum(Sales[Amount])
→ result
```

---

### Example 3: Iterator

**KAX:**

```morphogen
Sales[TotalRevenue] := SUMX(
    Sales,
    Sales[Price] * Sales[Quantity]
)
```

**GPU DAG:**

```
gpu.column_multiply(Sales[Price], Sales[Quantity])
→ gpu.agg_sum(result)
→ result
```

---

### Example 4: Time Intelligence

**KAX:**

```morphogen
Sales[YoY] :=
    Sales[TotalRevenue] -
    CALCULATE(
        Sales[TotalRevenue],
        SAMEPERIODLASTYEAR(Calendar[Date])
    )
```

**GPU DAG:**

```
// Current period
gpu.agg_sum(Sales[Amount]) → current_revenue

// Prior period
gpu.date_shift(Calendar[Date], -1, YEAR) → prior_dates
→ gpu.predicate(Calendar[Date], IN, prior_dates) → bitmap
→ gpu.bitmap_filter(Sales, bitmap)
→ gpu.agg_sum(Sales[Amount]) → prior_revenue

// Difference
gpu.scalar_subtract(current_revenue, prior_revenue)
→ result
```

---

## Optimization Passes

The KAX compiler applies optimization passes before lowering to GPU:

### 1. Constant Folding

```morphogen
Sales[Discounted] := Sales[Price] * 0.9
```

Folds constants during compilation.

---

### 2. Predicate Pushdown

```morphogen
CALCULATE(SUM(Sales[Amount]), Region = "USA", Year = 2024)
```

Pushes filters as early as possible:

```
gpu.predicate(Region, EQ, "USA") → bitmap1
gpu.predicate(Year, EQ, 2024) → bitmap2
gpu.bitmap_and(bitmap1, bitmap2) → combined_bitmap
gpu.bitmap_filter(Sales, combined_bitmap)
gpu.agg_sum(Sales[Amount])
```

---

### 3. Filter Fusion

Multiple filters on same table → combined bitmap.

```morphogen
CALCULATE(
    SUM(Sales[Amount]),
    Sales[Region] = "USA",
    Sales[Amount] > 100
)
```

Fuses into single bitmap:

```
gpu.predicate(Sales[Region], EQ, "USA") → bitmap1
gpu.predicate(Sales[Amount], GT, 100) → bitmap2
gpu.bitmap_and(bitmap1, bitmap2) → combined_bitmap
gpu.bitmap_filter(Sales, combined_bitmap)
```

---

### 4. Column Pruning

Only load columns referenced in expression.

```morphogen
Sales[AvgRevenue] := AVERAGE(Sales[Amount])
```

Only loads `Sales[Amount]`, not entire table.

---

### 5. Dictionary Encoding

Automatically encode low-cardinality columns.

```morphogen
CALCULATE(SUM(Sales[Amount]), Region = "USA")
```

Compiler recognizes `Region` is low-cardinality:

```
gpu.dict_encode(Sales[Region])
→ gpu.predicate_encoded(region_ids, dict["USA"])
→ gpu.bitmap_filter
```

---

## Type System

KAX supports Morphogen's type system:

```morphogen
// Primitive types
i32, i64, f32, f64, bool, string

// Date/time
date, datetime, time

// Columnar types (GPU-resident)
GpuColumn<T>
GpuTable

// Nullable
T?
```

Type inference:

```morphogen
Sales[TotalRevenue] := SUM(Sales[Amount])
// Inferred: f64 (if Sales[Amount] is f64)

Orders[Year] := YEAR(Orders[Date])
// Inferred: i32
```

---

## Error Handling

### Divide by Zero

```morphogen
Sales[MarginPct] := DIVIDE(Sales[Profit], Sales[Revenue])
```

`DIVIDE` returns `NULL` on divide-by-zero (safe version).

Alternatively:

```morphogen
Sales[MarginPct] := IF(
    Sales[Revenue] = 0,
    0,
    Sales[Profit] / Sales[Revenue]
)
```

---

### NULL Handling

```morphogen
SUM(column)       // Ignores NULLs
COUNT(column)     // Counts non-NULL
AVERAGE(column)   // Ignores NULLs
```

GPU compilation:

```
gpu.agg_sum(col)
→ Uses null bitmap to skip NULL values
```

---

## Cross-Domain Examples

### BI → Visualization

```morphogen
let sales_by_region = CALCULATE(
    SUM(Sales[Amount]),
    ALL(Sales),
    VALUES(Sales[Region])
)

viz.bar_chart(
    categories: sales_by_region.keys,
    values: sales_by_region.values,
    title: "Sales by Region"
)
```

---

### BI → Simulation

```morphogen
let avg_pressure = AVERAGE(SensorData[Pressure])
let std_pressure = STDEV.S(SensorData[Pressure])

physics.fluid_network(
    initial_pressure: avg_pressure,
    noise_amplitude: std_pressure,
    timesteps: 1000
)
```

---

### BI ↔ ML

```morphogen
// Feature engineering
Customers[LifetimeValue] := SUMX(
    FILTER(Orders, Orders[CustomerID] = Customers[CustomerID]),
    Orders[Amount]
)

Customers[Recency] := DATEDIFF(
    MAX(FILTER(Orders, Orders[CustomerID] = Customers[CustomerID])[Date]),
    TODAY(),
    DAY
)

// Train model
let model = ml.train_gpu(
    features: [Customers[LifetimeValue], Customers[Recency], ...],
    target: Customers[Churn],
    algorithm: XGBoost
)
```

---

### BI → Procedural Generation

```morphogen
let building_heights = CALCULATE(
    SUM(CityData[Population]),
    ALL(CityData),
    VALUES(CityData[District])
)

geometry.procedural_city(
    districts: building_heights.keys,
    heights: building_heights.values * 0.001,
    style: "modern"
)
```

---

## Implementation Roadmap

### Phase 1: Parser
- Lexer & tokenizer
- Expression parser
- AST generation

---

### Phase 2: Type System & Binding
- Type inference
- Column binding
- Scope resolution
- Error reporting

---

### Phase 3: Core Functions
- Aggregations (`SUM`, `COUNT`, `AVG`, etc.)
- Arithmetic operators
- Comparison operators
- Logical operators

---

### Phase 4: Context Manipulation
- `CALCULATE`
- `FILTER`
- `ALL` / `ALLEXCEPT`
- Filter context tracking

---

### Phase 5: Iterators
- `SUMX`, `AVERAGEX`, `COUNTX`
- Row context transitions

---

### Phase 6: Time Intelligence
- Date extraction (`YEAR`, `MONTH`, etc.)
- `SAMEPERIODLASTYEAR`
- `PREVIOUSMONTH` / `PREVIOUSQUARTER`
- `DATESINPERIOD`

---

### Phase 7: GPU Compilation
- Lowering to GPU ops
- Optimization passes (predicate pushdown, filter fusion, etc.)
- Operator graph generation
- CUDA kernel selection

---

### Phase 8: Cross-Domain Integration
- BI → Viz pipelines
- BI → Simulation
- BI ↔ ML
- BI → Procedural generation

---

## Performance Targets

| Expression Type | Target Latency | Comparison |
|----------------|---------------|------------|
| Simple aggregation | < 0.5 ms | 5x faster than VertiPaq |
| Filtered measure | < 1 ms | 10x faster than DAX |
| Iterator (SUMX) | < 2 ms | 15x faster than row-by-row |
| Time intelligence | < 3 ms | 20x faster than CPU DAX |

---

## Future Extensions

### Advanced Features
- Calculated tables
- Many-to-many relationships
- Bi-directional filtering
- Hierarchies & drill-down
- Dynamic format strings

### Performance
- Query result caching
- Incremental refresh
- Pre-aggregations
- Materialized views

### Interoperability
- Import DAX expressions directly
- Export to Power BI
- ODBC/JDBC query interface

---

## Summary

**KAX (Morphogen Analytical eXpressions)** is:

✅ **DAX-inspired** — Familiar semantic model for BI users
✅ **GPU-compiled** — Expressions lower to GPU operator graphs
✅ **Domain-composable** — Integrates with all Morphogen domains
✅ **Declarative** — Engine optimizes execution automatically
✅ **High-performance** — 5-20x faster than traditional BI engines

KAX + BIDomain makes Morphogen the first **Computational BI Engine** —
fusing analytics with simulation, ML, visualization, and physics in one unified platform.

---

## Related Documents

- **bi-domain.md:** BI domain architecture and operators
- **../adr/007-gpu-first-domains.md:** GPU-first domains decision
- **operator-registry.md:** Operator registration and extensibility
- **../architecture/gpu-mlir-principles.md:** GPU lowering and MLIR integration

---

**End of Specification**
