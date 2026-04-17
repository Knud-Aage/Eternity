The MasterSolverPBP class is the high-performance "Piece-By-Piece" (PBP) engine of your solver. Unlike the Macro-Tile approach which tries to solve 4x4 blocks, this class places pieces one at a time in a specific sequence. It uses a hybrid architecture that combines CPU multi-threading for "broad" search and GPU acceleration for "deep" exploration.
Here is a breakdown of how the key components work together:
1. Build Orders: Typewriter vs. Spiral
The efficiency of a backtracking solver is heavily dependent on the order in which cells are filled. This is because the earlier you can highly constrain a cell (by having neighbors already placed), the faster you can prune invalid branches.
•
Typewriter Order (generateTypewriterOrder): This is the standard linear approach. It fills the board from index 0 to 255 (left-to-right, top-to-bottom). While simple, its weakness is that pieces in the middle of a row are often only constrained by their West neighbor, leading to a high branching factor.
•
Spiral Order (generateSpiralOrder): This starts at the top-left corner (0,0) and spirals inward. By using a spiral, the solver tends to complete edges and then "wrap" around the existing structure. This often ensures that new pieces are constrained by at least two neighbors (e.g., North and West) much sooner than in a linear search, significantly reducing the number of valid candidates to check at each step.
2. The CPU Role: "Wide" Seed Generation
The CPU doesn't try to solve the entire puzzle. Instead, it acts as a "Seed Generator."
•
Multi-threading: The solver detects your CPU cores and creates a thread pool. Each thread runs a SearchWorker.
•
Handoff Depth: Each worker performs a standard backtrack until it reaches the handoffDepth (e.g., 50 or 70 pieces placed).
•
Forward Checking: To keep the CPU search efficient, it uses a "Lookahead" optimization. Before placing a piece, it checks if the empty cells directly below or to the right still have any matching pieces left in the inventory. If placing a piece "dooms" a neighbor, it prunes that branch immediately.
•
Seed Collection: Once a worker reaches the handoffDepth, it stops, clones the current board state, and adds it to the gpuSeedBoards queue. These partial boards are the "seeds" that will be given to the GPU.
3. The GPU Role: "Deep" Parallel Search
The GPU is where the "brute-force" happens. While a CPU might have 16 or 32 threads, your GPU has thousands of cores.
•
Massive Parallelism: The runGpuHandoff method takes thousands of seeds (the targetBatchSize) and uploads them to the GPU. Every CUDA thread takes one seed and tries to finish the puzzle from that specific starting point.
•
Inventory Tracking: In the CUDA kernel (SolveEternityKernel.cu), the inventory is tracked using bit-masks (unsigned long long inventoryMask[4]). Checking if a piece is available is a lightning-fast bitwise operation rather than an array lookup.
•
The Radar Leash: This is a safety mechanism. If a GPU thread wanders too deep into a branch without finding a solution (the radarLimit), it forces a backtrack. This ensures the GPU doesn't get stuck in a single massive, useless branch.
4. Coordination and Resilience
The class includes several advanced features to ensure the solver doesn't get "stuck":
•
Base Camp Logic: The deepestStep tracks the best progress ever made. The solver periodically locks a "Base Camp" (e.g., 30 steps behind the high score). If the current search space is exhausted, it retreats to this base camp to try a different direction.
•
Extinction Events: If the GPU reports that a huge percentage of seeds were "Dead on Arrival" (couldn't place more than 5 pieces), the solver triggers a Poisoned Base Camp exception. It realizes the current path is a dead-end, retreats significantly further back, and tries again.
•
Stagnation Tracking: If no new high score is found for a set amount of time (e.g., 20 minutes), the solver triggers an Autonomous Deep Extinction, forcing a retreat to a much earlier state to ensure search diversity.