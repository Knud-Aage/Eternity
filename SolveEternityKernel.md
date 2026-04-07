This CUDA kernel, SolveEternityKernel.cu, is the high-performance heart of your solver. It uses an iterative backtracking approach to search for the puzzle solution, offloading millions of calculations to the GPU's parallel cores.
Here is a breakdown of how the kernel operates:
1. Bit-Packed Color Extraction
At the top of the file, you have several inline helper functions: getNorth, getEast, getSouth, and getWest. These functions use bitwise shifts and masks to extract the four 8-bit color IDs stored within a single 32-bit integer. This packing is critical for performance because it allows the GPU to move piece data as a single 4-byte word rather than four separate values.
2. Adjacency and Border Logic (matches)
The matches function is the "rule book" for the puzzle:
•
Neighbor Matching: It compares the colors of a candidate piece against the required colors from neighbors (n_req, e_req, etc.). If a requirement is 255 (your wildcard), it ignores that edge.
•
The Border Patrol: It enforces that only "Grey" edges (ID 0) can touch the absolute boundaries of the 16x16 grid. Specifically, it ensures that if a piece is not on a specific edge (e.g., row != 0), its corresponding side (North) cannot be color 0.
3. The Core Solver Engine (solvePBP)
The main kernel, solvePBP, is designed so that every GPU thread works on a unique "seed" or partial board configuration provided by the Java side.
•
Inventory Management: It uses an unsigned long long inventoryMask[4] (totaling 256 bits) to track which physical pieces are currently used on the board. This is much faster than searching an array.
•
Iterative Backtracking: Instead of using recursion (which is expensive on GPUs), the kernel uses a while loop and a pieceStack to manage the search.
◦
If a piece fits at the current pos, it marks the inventory mask, saves the current index to the stack, and moves to pos++.
◦
If no piece fits, it resets the stack for that position, moves to pos--, and "returns" the previous piece to the inventory mask.
4. Pruning and Optimization
The kernel includes a significant optimization called the "Doomed" check. Every time a row is completed (when (pos + 1) % 16 == 0), the kernel performs a look-ahead:
1.
It counts the colors exposed on the "South" boundary of the completed row.
2.
It counts the total available colors remaining in the unused inventory.
3.
If any required color count exceeds the available count, the kernel knows this branch is impossible ("doomed") and triggers an immediate backtrack, saving millions of useless iterations.
5. Global State and Output
Since thousands of threads are running simultaneously, the kernel uses Atomic Operations to communicate results back to Java:
•
Solution Found: If a thread reaches pos == 256, it uses atomicExch on d_solvedFlag. The first thread to find a solution "locks" the flag and writes its board to d_solution.
•
High Score Tracking: Even if a solution isn't found, the kernel tracks progress. It compares its localMax (deepest search depth) against a global d_gpuHighScore using atomicCAS (Compare-And-Swap). If it found a new best configuration, it updates the global best board so you can see the progress in your GUI.