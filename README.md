# Eternity II Solver

A world-class, high-performance Eternity II puzzle solver. This engine utilizes a hybrid metaheuristic approach, combining evolutionary algorithms on the CPU with massively parallel backtracking kernels on the GPU to navigate the $10^{600}$ search space of the 256-piece puzzle.

## 🏗 System Architecture & Hardware
The engine is designed to adapt to your hardware capabilities:
- **Hybrid CPU/GPU Mode:** The "Gold Standard" for this solver. The CPU handles the "brains"—genetic selection, scoring, and strategy—while the NVIDIA GPU (via CUDA) acts as the "muscle," performing billions of placement checks per second.
- **CPU-Only Mode:** A fully functional fallback that utilizes multi-threaded parallel DFS for environments without CUDA support.

## 🔄 Search Strategies & Build Orders
The order in which the board is filled changes the "shape" of the constraints. The solver supports:
- **Spiral (Advanced):** Starts from the edges and wraps inward. This builds a "frame" early, which significantly constrains the internal pieces and reduces the branching factor of the inner 14x14 grid.
- **Typewriter (Layered):** A standard top-left to bottom-right approach. While simpler, it is highly effective for discovering specific row-based patterns and is often used for "base camp" generation.

## 🚀 The Solving Pipeline (3-Phase Execution)

### Phase 1: Evolutionary Seed Generation (CPU)
The solver begins by generating thousands of "seeds"—valid partial board configurations (usually 40-60 pieces).
- **Weighted Selection:** The `SeedSelector` uses a heuristic fitness function to rank boards. It rewards pieces placed in high-value positions and applies a **Danger Penalty** to boards with "trapped" empty slots (slots with 3 or 4 filled neighbors) that are statistically impossible to satisfy.
- **Genetic Refinement:** Like a genetic algorithm, the solver uses **Elitism** to keep the best boards, **Mutation** to swap internal pieces and escape local optima, and **Random Exploration** to maintain population diversity.

### Phase 2: Massively Parallel Deep Search (GPU)
The best seeds are offloaded to the GPU. Each CUDA thread takes a seed and performs a deep, high-speed backtracking search.
- **Radar Leash:** An optimization that prevents threads from wandering too deep into "dead" sub-trees. If a thread doesn't find a significant improvement within a set distance from its handoff point, it is forced to backtrack and try a different branch.

### Phase 3: The Surgeon - LNS Repair (Metaheuristic)
When progress stalls (usually above 210 pieces), the solver invokes **The Surgeon**. This implements a **Large Neighborhood Search (LNS)**:
- **Punching Holes:** The Surgeon identifies "conflict zones" where piece edges are mismatched or difficult to satisfy.
- **Targeted vs. Random:** It strategically "punches holes" in the board—removing pieces in high-conflict areas while leaving the rest of the board intact.
- **Tabu Search:** To prevent the solver from falling back into the same local optimum, it uses a **Tabu Tenure** system, marking recently modified pieces as "off-limits" for removal.

## 🧠 Core Heuristics & Optimizations
- **Position Weighting:** Pieces closer to the center are weighted higher, as they have more constraints (4 edges) than edge pieces.
- **Lookahead Pruning:** CPU workers check the compatibility of future neighbors *before* placing a piece, failing fast on invalid paths.
- **Bit-Packed Data:** Pieces and board states are bit-packed into 32-bit integers, allowing the entire compatibility check to happen in a single CPU/GPU register cycle.
- **Cloud Sync:** Integrated with Google Drive API to synchronize high-score records and checkpoints across different machines.

## Getting Started

### Prerequisites
*   Java 17 or higher.
*   NVIDIA GPU with CUDA Toolkit (Required for GPU mode).
*   JCuda libraries for Java-CUDA binding.

### Running
1.  Ensure `pieces.csv` is in the root directory.
2.  Download/Place piece pattern images in the `/Assets` folder for visualization.
3.  Compile: `mvn clean install`.
4.  Run `dk.puzzle.core.Main`.
5.  Select your strategy in the **Startup Dialog**:
    *   Choose **Spiral** for advanced constraint satisfaction.
    *   Toggle **GPU Accelerated** for maximum performance.

## 📊 Outputs & Monitoring
- **Real-time GUI:** Watch the solver "breathe" as it explores different branches.
- **Local Records:** Strategy-specific folders (e.g., `/records/SPIRAL/`) contain PNG snapshots and CSV data of every new high score.
- **Cloud Sync:** Checkpoints and records are automatically mirrored to Google Drive.
- **Smart Resume:** The engine automatically detects the highest-score checkpoint in your profile folder and resumes from there upon startup.

---
*Developed for the Eternity II Research Project.*
