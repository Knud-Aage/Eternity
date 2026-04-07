do javadoc for # Eternity II Solver

High-performance Eternity II puzzle solver written in Java with GPU acceleration.

## Features

### Solving Strategies
The solver supports two distinct algorithmic approaches:

1.  **Divide and Conquer (Macro-Tile Strategy)**:
    *   Splits the 16x16 board into a 4x4 grid of "Macro-Tiles" (each is a 4x4 piece block).
    *   Solves the perimeter of each macro-tile first, then validates internal matches using the GPU/CPU validator.
    *   Significantly reduces the branching factor of the search tree.

2.  **Piece-by-Piece (Linear DFS)**:
    *   A classic depth-first search approach that places pieces one by one.
    *   Useful for testing and smaller sub-puzzles.

### Hardware Acceleration
*   **GPU Validation**: Leverages NVIDIA GPUs via **JCuda** to validate millions of 4x4 macro-tile candidates simultaneously.
*   **CPU Fallback**: If no compatible GPU is detected, the solver automatically switches to a multi-threaded CPU validator using Java Parallel Streams to maximize performance on all available cores.

### Advanced Heuristics
*   **Color Frequency Scoring**: Prioritizes placing pieces that expose common colors to the unsolved areas of the board.
*   **Pool Partitioning**: Automatically separates pieces into Corner, Edge, and Interior pools to eliminate illegal search branches.
*   **Diagonal Wavefront Scan**: Fill order optimized to maximize constraints at every step.

## Getting Started

### Prerequisites
*   Java 17 or higher.
*   (Optional) NVIDIA GPU with latest drivers and CUDA Toolkit for GPU acceleration.

### Running
1.  Ensure `pieces.csv` is in the root directory.
2.  Compile the project using Maven: `mvn clean install`.
3.  Run the `Main` class.
4.  The GUI will open, showing a real-time visualization of the solving process.

## Output
*   **solution.txt**: A text-based map of the final 16x16 solution.
*   **solution.png**: A high-resolution image rendering of the solved puzzle.
