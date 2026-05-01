# Eternity II Solver

High-performance Eternity II puzzle solver written in Java with GPU acceleration.

## Features

### Solving Strategies
The solver employs a hybrid Piece-By-Piece (PBP) approach that combines CPU exploration with massive GPU parallelization:

1.  **Autonomous PBP Search**:
    *   Uses the CPU to generate thousands of "seeds" (partial board states) at a specific handoff depth.
    *   Features adaptive "Extinction Events" and "Evolution Leaps" to navigate around structural dead-ends.
    *   Supports multiple build orders: **Spiral** (inward-wrapping constraints) and **Typewriter** (linear).

2.  **LNS Repair Mode (The Surgeon)**:
    *   A Large Neighborhood Search (LNS) variant for the late-game.
    *   Targets "high-conflict" areas with edge mismatches and "punches holes" in the board.
    *   Uses the GPU to solve targeted holes (Swiss Cheese boards) to bridge the gap to a 256-piece solution.

### Hardware Acceleration
*   **CUDA Kernel Integration**: Custom-written CUDA kernels (`SolveEternityKernel.cu`) handle the heavy backtracking and repair logic.
*   **Radar Leash**: A look-ahead optimization that forces backtracking in GPU threads if a solution isn't found within a certain distance from the handoff point, preventing thread divergence and stagnation.
*   **Multi-threaded Seed Generation**: Leverages all available CPU cores to explore the wide search space before offloading deep dives to the GPU.

### Advanced Heuristics
*   **Lookahead Optimization**: CPU workers check secondary neighbor viability before placing pieces, significantly pruning the search tree.
*   **Structural Diversity Filter**: Prevents the GPU from processing redundant search paths by hashing board configurations.
*   **Bit-Masked Inventory**: Rapid piece availability checking using 256-bit masks across GPU registers.

## Getting Started

### Prerequisites
*   Java 17 or higher.
*   NVIDIA GPU with latest drivers and CUDA Toolkit for GPU acceleration.
*   JCuda libraries for Java-CUDA binding.

### Running
1.  Ensure `pieces.csv` is in the root directory.
2.  Compile the project using Maven: `mvn clean install`.
3.  Run the `Main` class.
4.  Use the "Director Controls" in the GUI to adjust extinction thresholds, radar limits, and manual overrides in real-time.

## Output
*   **records/**: Strategy-specific folders containing high-score screenshots (.png) and board data (.csv).
*   **checkpoint.dat**: Automatic serialization of the current working state to allow resumption after a shutdown.
