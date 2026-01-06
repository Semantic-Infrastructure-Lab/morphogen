# SPEC: BI Domain for GPU-Native Analytics

> 🔬 **Implementation Status: RESEARCH SPECIFICATION**
> This is a proposed feature for future Morphogen versions. Not yet implemented.

**Version:** 1.0
**Status:** Proposed (post-v1.0)
**Last Updated:** 2025-11-15
**Related:** ../adr/007-gpu-first-domains.md, kax-language.md

---

## Overview

This document specifies the **BIDomain** — Morphogen's first GPU-first computational domain that brings Business Intelligence, analytical query engines, and semantic modeling into the Morphogen ecosystem.

### Why BIDomain?

BI represents a paradigm shift for Morphogen because:

1. **GPU is not just an accelerator — it defines the semantics** — Dictionary encoding, bit-packing, and warp-level operations are intrinsic to the algorithm
2. **Columnar compression is GPU-native** — VertiPaq-style encoding requires GPU memory layouts
3. **Cross-domain analytics** — BI can drive simulation parameters, visualizations, ML pipelines, and procedural generation
4. **Computational BI** — Unlike Power BI or Tableau, Morphogen BI integrates with physics, rendering, and optimization
5. **Declarative semantic layer** — KAX (Morphogen Analytical eXpressions) provides DAX-like modeling on GPU primitives

### Applications

**Business & Analytics:**
- GPU-accelerated OLAP cubes
- Real-time dashboards with millions of rows
- Multidimensional modeling
- Time intelligence and calculated measures
- Star schema analytics

**Scientific Computing:**
- High-dimensional data exploration
- Simulation parameter analytics
- GPU-accelerated statistical analysis
- Cross-domain data pipelines

**Creative & Procedural:**
- Data-driven procedural generation
- Visualization from analytics
- Interactive exploration of simulation results
- BI-guided design optimization

**Hybrid Workflows:**
- BI → GPU visualization (charts, heatmaps, 3D plots)
- BI → Simulation parameterization
- BI ↔ ML feature engineering
- BI ↔ Physics (analyze results, drive parameters)

---

## Comparison with Existing BI Engines

### VertiPaq (Power BI, SSAS Tabular)

**What VertiPaq Does Well:**
- Extreme columnar compression (dictionary + RLE + bit-packing)
- Segment metadata for query pruning
- Fast vectorized aggregations
- Tight DAX integration
- Multidimensional semantic modeling

**What VertiPaq Cannot Do:**
- ❌ GPU acceleration
- ❌ Custom operators or procedural compute
- ❌ Integration with simulation, ML, or rendering
- ❌ Open architecture
- ❌ Extensible type system

**Morphogen BI Advantages:**
- ✅ GPU-first compression & scan kernels
- ✅ Custom operator registration
- ✅ Composable with all Morphogen domains
- ✅ Open, extensible architecture
- ✅ DAX-like semantics (KAX) on GPU primitives

---

### DuckDB / Arrow / Polars / DataFusion

**What DataFrame Engines Do Well:**
- Columnar in-memory processing
- Vectorized CPU operators
- Strong I/O performance
- SQL query engines
- Embedded analytics

**What DataFrame Engines Lack:**
- ❌ GPU-native operators (optional at best)
- ❌ BI semantic modeling (DAX-style measures)
- ❌ Multidimensional analytics
- ❌ Integration with simulation or computational domains
- ❌ Unified declarative operator system

**Morphogen BI Advantages:**
- ✅ GPU-native dictionary encoding, compressed scans
- ✅ KAX semantic layer for measures and calculated columns
- ✅ Cross-domain composability
- ✅ Unified operator registry and scheduler
- ✅ Automatic GPU lowering and optimization

---

### RAPIDS / cuDF / OmniSciDB

**What GPU DataFrame Engines Do Well:**
- True GPU columnar processing
- Rapid filtering, grouping, joins
- Familiar dataframe API
- Vectorized GPU kernels

**What GPU DataFrame Engines Lack:**
- ❌ DAX-style semantic model
- ❌ Multidimensional modeling
- ❌ Integration with simulation or physics
- ❌ Universal declarative layer
- ❌ Composability with other computational domains

**Morphogen BI Advantages:**
- ✅ KAX expression language (DAX-like on GPU)
- ✅ Unified operator model across all domains
- ✅ Composable workflows (BI + graphics + ML + physics)
- ✅ Automatic scheduling and optimization
- ✅ Domain-specific operator registration

---

## The GPU-First Paradigm Shift

Until BIDomain, Morphogen's GPU abstraction was:

> "Operators auto-lower to GPU when beneficial for performance."

Examples: `fft`, `noise`, `simulation.step`, `automaton.update`

This worked because:
- Operators behaved identically on CPU and GPU
- GPU was an accelerator, not a semantic requirement
- Device-agnostic execution was possible

### BI Changes Everything

A VertiPaq-style engine requires:

1. **Dictionary encoding** — String → Integer ID mapping
2. **Bit-packing** — Compressed integer storage
3. **Segmented columnar storage** — Sorted, partitioned data
4. **Warp-level operations** — Parallel scan, prefix sum
5. **Bitmap filters** — Compressed predicate evaluation
6. **GPU-resident metadata** — Segment boundaries, min/max stats

These are **GPU-specific data structures**.

Running them on CPU:
- ❌ Different semantics
- ❌ Different memory layout
- ❌ Not just slower — fundamentally different

### GPU-First Domains

BIDomain introduces:

**Property:** Operators require GPU memory formats
**Consequence:** CPU fallback may be impossible or limited
**Implication:** GPU is the internal representation, not an optimization

Examples of GPU-first operators:

```morphogen
gpu.dict_encode      // Dictionary encoding
gpu.segmented_groupby // Segmented aggregation
gpu.bitmap_filter    // Compressed filtering
gpu.sort_radix       // Radix sort
gpu.prefix_sum_warp  // Warp-level scan
gpu.hash_join_segmented // GPU-native join
```

These operators have **no meaningful CPU equivalents** in the traditional Morphogen sense.

---

## BIDomain Architecture

### Domain Definition

```morphogen
domain: BI
device: gpu
```

The `device: gpu` qualifier indicates:
- GPU-backed operators
- GPU-resident columnar buffers
- GPU-only execution by default

CPU-backed BI is considered separately as `BICPU` domain (if needed).

---

## Data Model

### Base Types

```morphogen
// Core GPU columnar types
type GpuColumn<T> {
    data: GpuBuffer<T>,
    count: i64,
    nulls: GpuBitmap?
}

type GpuDictEncodedColumn<T> {
    dictionary: GpuColumn<T>,
    indices: GpuColumn<i32>,
    count: i64
}

type GpuBitmap {
    bits: GpuBuffer<u32>,
    count: i64
}

type GpuSegment {
    start: i64,
    end: i64,
    min: Scalar,
    max: Scalar,
    null_count: i64
}

type GpuTable {
    columns: Map<String, GpuColumn<Any>>,
    row_count: i64,
    segments: GpuColumn<GpuSegment>?
}

type GpuGroup<K, V> {
    keys: GpuColumn<K>,
    values: GpuColumn<V>,
    groups: GpuColumn<GpuSegment>
}
```

### Properties

All columns are:
- **Columnar** — Values stored contiguously
- **Compressed** — Dictionary encoded where beneficial
- **GPU-resident** — Primary storage on device
- **Nullable** — Optional bitmap for NULL tracking

Tables are:
- Collections of aligned columns
- Row count invariant across columns
- Optional segmentation metadata
- GPU-resident by default

---

## Core Operators

### Compression & Encoding

```morphogen
// Dictionary encode a column
op gpu.dict_encode<T>(
    col: GpuColumn<T>
) -> GpuDictEncodedColumn<T>

// Decode back to values
op gpu.dict_decode<T>(
    encoded: GpuDictEncodedColumn<T>
) -> GpuColumn<T>

// Bit-pack integer column
op gpu.bitpack(
    col: GpuColumn<i64>,
    bits_per_value: i32
) -> GpuColumn<u8>
```

---

### Filtering

```morphogen
// Create bitmap from predicate
op gpu.predicate<T>(
    col: GpuColumn<T>,
    op: ComparisonOp,
    value: T
) -> GpuBitmap

// Filter table by bitmap
op gpu.bitmap_filter(
    bitmap: GpuBitmap,
    table: GpuTable
) -> GpuTable

// Combine bitmaps (AND, OR, NOT)
op gpu.bitmap_combine(
    a: GpuBitmap,
    b: GpuBitmap,
    op: BitmapOp
) -> GpuBitmap
```

---

### Aggregation

```morphogen
// Segmented group-by aggregation
op gpu.segmented_groupby<K, V>(
    keys: GpuColumn<K>,
    values: GpuColumn<V>,
    agg: AggOp
) -> GpuGroup<K, V>

// Aggregate functions
enum AggOp {
    Sum,
    Count,
    Avg,
    Min,
    Max,
    StdDev,
    Variance
}

// Simple aggregate over entire column
op gpu.agg_sum(col: GpuColumn<f64>) -> f64
op gpu.agg_count(col: GpuColumn<Any>) -> i64
op gpu.agg_avg(col: GpuColumn<f64>) -> f64
op gpu.agg_min<T>(col: GpuColumn<T>) -> T
op gpu.agg_max<T>(col: GpuColumn<T>) -> T
```

---

### Sorting

```morphogen
// GPU radix sort (stable)
op gpu.sort_radix<T>(
    col: GpuColumn<T>
) -> GpuColumn<T>

// Segmented sort
op gpu.sort_segmented<T>(
    col: GpuColumn<T>,
    segments: GpuColumn<GpuSegment>
) -> GpuColumn<T>

// Sort index generation
op gpu.argsort<T>(
    col: GpuColumn<T>
) -> GpuColumn<i64>
```

---

### Joins

```morphogen
// Hash join on GPU
op gpu.hash_join<K, V1, V2>(
    left: GpuTable,
    right: GpuTable,
    left_key: String,
    right_key: String,
    join_type: JoinType
) -> GpuTable

enum JoinType {
    Inner,
    Left,
    Right,
    Full
}

// Segmented join (for star schemas)
op gpu.hash_join_segmented<K, V1, V2>(
    fact: GpuTable,
    dimension: GpuTable,
    key: String
) -> GpuTable
```

---

### Scanning & Reduction

```morphogen
// Warp-level prefix sum
op gpu.scan_warp(
    col: GpuColumn<i32>
) -> GpuColumn<i32>

// Block-level prefix sum
op gpu.scan_block(
    col: GpuColumn<i32>
) -> GpuColumn<i32>

// Global prefix sum
op gpu.prefix_sum(
    col: GpuColumn<i32>
) -> GpuColumn<i32>
```

---

### Measure Evaluation

```morphogen
// Evaluate a KAX measure expression
op gpu.measure(
    table: GpuTable,
    expr: KAXExpression,
    context: FilterContext
) -> GpuScalar

// Evaluate calculated column
op gpu.calc_column(
    table: GpuTable,
    expr: KAXExpression,
    context: RowContext
) -> GpuColumn<Any>
```

---

## Cross-Domain Integration

### BI → Visualization

```morphogen
// Generate GPU heatmap from BI aggregation
let sales_by_region = gpu.segmented_groupby(
    keys: Sales[Region],
    values: Sales[Amount],
    agg: Sum
)

let heatmap = viz.gpu_heatmap(
    data: sales_by_region,
    colormap: Viridis
)

render.gpu(heatmap)
```

---

### BI → Simulation

```morphogen
// Use BI to parameterize physics simulation
let avg_temp = gpu.agg_avg(SensorData[Temperature])
let std_temp = gpu.agg_stddev(SensorData[Temperature])

let thermal_sim = physics.thermal_ode(
    initial_temp: avg_temp,
    noise_level: std_temp,
    timestep: 0.01
)
```

---

### BI ↔ ML

```morphogen
// Feature engineering with BI, then ML training
let features = gpu.calc_column(
    table: Customers,
    expr: [
        LifetimeValue := SUM(Orders[Amount]),
        AvgOrderSize := DIVIDE(SUM(Orders[Amount]), COUNT(Orders)),
        Recency := DATEDIFF(TODAY(), MAX(Orders[Date]))
    ]
)

let model = ml.train_gpu(
    features: features,
    target: Customers[Churn],
    algorithm: XGBoost
)
```

---

### BI → Procedural Generation

```morphogen
// Data-driven geometry generation
let city_populations = gpu.segmented_groupby(
    keys: Cities[Name],
    values: Cities[Population],
    agg: Sum
)

let buildings = geometry.procedural_buildings(
    positions: Cities[Location],
    heights: city_populations.values,
    scale: 0.001
)

render.gpu(buildings)
```

---

## KAX: Morphogen Analytical eXpressions

See **kax-language.md** for full details.

KAX is Morphogen's DAX-like semantic expression language:

### Measures

```morphogen
Sales[TotalAmount] := SUM(Sales[Amount])

Sales[MarginPct] := DIVIDE(Sales[Profit], Sales[Revenue])
```

### Calculated Columns

```morphogen
Orders[DayOfWeek] := WEEKDAY(Orders[Date])
```

### Filter Context

```morphogen
Sales[2024Revenue] := CALCULATE(
    SUM(Sales[Amount]),
    Year = 2024
)
```

### Time Intelligence

```morphogen
Sales[YoY] :=
    Sales[TotalAmount] -
    CALCULATE(
        Sales[TotalAmount],
        SAMEPERIODLASTYEAR(Calendar[Date])
    )
```

### GPU Compilation

KAX expressions compile into GPU operator graphs:

```morphogen
CALCULATE(SUM(Sales[Amount]), Year = 2024)
```

Compiles to:

```
gpu.dict_encode(Calendar[Year])
→ gpu.predicate(Year, EQ, 2024)
→ gpu.bitmap_filter
→ gpu.agg_sum(Sales[Amount])
```

---

## Execution Pipeline

```
KAX Expression
      ↓ parse
Expression Tree
      ↓ bind columns
Bound Expression
      ↓ optimize
Operator Graph
      ↓ schedule
GPU Execution Plan
      ↓ lower
CUDA Kernels
      ↓ execute
GPU Results
```

The Morphogen scheduler orchestrates this pipeline just like it does for physics, rendering, or audio tasks.

---

## Why Morphogen Should Own BI

Morphogen is uniquely positioned because:

1. **Composability across domains** — BI integrates with simulation, ML, rendering, physics
2. **GPU-centered execution** — BIDomain leverages Morphogen's GPU infrastructure
3. **Declarative + procedural blending** — KAX + custom operators
4. **Operator registration** — Extensible BI operators
5. **No legacy constraints** — Clean slate design

### What This Enables

- **Computational BI** — BI fused with simulation, optimization, procedural generation
- **Interactive visualization** — Measures drive GPU rendering directly
- **Hybrid analytics** — BI ↔ ML ↔ physics in one pipeline
- **Real-time dashboards** — GPU-accelerated aggregations
- **Data-driven design** — BI guides procedural modeling

This is not Power BI, not Tableau, not DuckDB, not RAPIDS —
**Morphogen becomes a Computational BI Engine.**

---

## Implementation Phases

### Phase 1: Core GPU Data Structures
- `GpuColumn<T>`
- `GpuTable`
- `GpuDictEncodedColumn<T>`
- GPU memory allocators
- Columnar metadata

**Deliverable:** Basic GPU columnar types

---

### Phase 2: Primitive GPU Operators
- Prefix scans (`gpu.scan_warp`, `gpu.prefix_sum`)
- Bitmap filters (`gpu.predicate`, `gpu.bitmap_filter`)
- Radix sorts (`gpu.sort_radix`)
- Dictionary encode/decode
- Warp-level reductions
- Segment metadata

**Deliverable:** Low-level GPU kernels

---

### Phase 3: BI Operators & Aggregations
- `gpu.groupby`
- `gpu.agg_sum`, `gpu.agg_count`, `gpu.agg_avg`, etc.
- `gpu.join` (hash join)
- Column projections
- Basic measure infrastructure

**Deliverable:** Functional BI operators

---

### Phase 4: KAX Expression Engine
- Parser (KAX syntax)
- Expression tree
- Column binding
- Optimization passes (predicate pushdown, filter fusion)
- Lowering to GPU ops

**Deliverable:** DAX-like semantic layer

---

### Phase 5: Cross-Domain Integration
- BI → Visualization pipelines
- BI → Simulation parameterization
- BI ↔ ML feature engineering
- BI → Procedural generation
- Example workflows

**Deliverable:** Hybrid computational BI workflows

---

## Performance Targets

| Operation | Target Throughput | Comparison |
|-----------|------------------|------------|
| Dictionary encode | > 10 GB/s | 10x DuckDB CPU |
| Grouped aggregation | > 5 GB/s | 20x Polars CPU |
| Bitmap filter | > 20 GB/s | 30x Arrow CPU |
| Hash join | > 3 GB/s | 15x DataFusion CPU |
| Measure evaluation (KAX) | < 1ms latency | 5x VertiPaq |

---

## Future Directions

### Advanced Compression
- Run-length encoding (RLE)
- Frame-of-reference encoding
- Bit-shuffle + LZ4
- Adaptive compression selection

### Distributed Execution
- Multi-GPU data parallelism
- Cross-node joins
- Distributed aggregations

### Advanced Semantics
- Hierarchies & drill-down
- Calculated tables
- Many-to-many relationships
- Bi-directional filtering

### Query Optimization
- Cost-based optimization
- Statistics-driven planning
- Adaptive execution
- Query caching

---

## Summary

The **BIDomain** represents a fundamental shift in Morphogen's architecture:

✅ **First GPU-first domain** — GPU is semantic, not just acceleration
✅ **Columnar GPU data structures** — Dictionary encoding, bitmaps, segments
✅ **KAX semantic layer** — DAX-like expressions on GPU primitives
✅ **Cross-domain composability** — BI integrates with all Morphogen domains
✅ **Computational BI** — BI fused with simulation, ML, rendering, physics

With BIDomain, Morphogen becomes a **universal computational platform** that handles:

- Business Intelligence & Analytics
- GPU-accelerated simulation
- Real-time visualization
- Machine learning
- Procedural generation
- Physics & optimization
- Audio & DSP

**All with one unified architecture.**

---

## Related Documents

- **../adr/007-gpu-first-domains.md:** GPU-First Domains Architecture Decision
- **kax-language.md:** KAX Expression Language Specification
- **operator-registry.md:** Operator registration and extensibility
- **scheduler.md:** Morphogen scheduling and execution model
- **../architecture/gpu-mlir-principles.md:** GPU lowering and MLIR integration

---

**End of Specification**
