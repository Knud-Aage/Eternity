using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace EternityII_Solver
{
    class Program
    {
        // Original value 64 (Blackwood's own machine). This machine has 28
        // logical processors -- kept his own "-1 for us" reservation logic
        // (Parallel.For(1, number_virtual_cores, ...) always uses
        // number_virtual_cores-1 threads) but matched the constant to actual
        // hardware instead of oversubscribing. Hardware-matching only, no
        // change to search logic/budgets/candidate selection.
        //
        // Overridable via ETERNITY_VIRTUAL_CORES: the live production solver
        // (unmodified, 10-break schedule) is already using up to 27 of this
        // machine's 28 threads. This copy runs alongside it, not instead of
        // it, so a real drop_239 trial needs a deliberately smaller share
        // rather than the full 28.
        static int number_virtual_cores = 28;

        // 2026-08-16 break-tuning harness. node_cap replaces the hardcoded
        // 50-billion literal SolvePuzzle() used to check against -- same
        // variable, just settable, so short comparison runs don't have to
        // wait for a production-length attempt to time out on its own.
        static long node_cap = 50_000_000_000L;

        static void Main()
        {
            string runLabel = Environment.GetEnvironmentVariable("ETERNITY_RUN_LABEL") ?? "default";
            string nodeCapEnv = Environment.GetEnvironmentVariable("ETERNITY_NODE_CAP");
            if (!string.IsNullOrWhiteSpace(nodeCapEnv)) node_cap = long.Parse(nodeCapEnv);
            string secondsEnv = Environment.GetEnvironmentVariable("ETERNITY_TUNE_SECONDS");
            long? tuneSeconds = string.IsNullOrWhiteSpace(secondsEnv) ? (long?)null : long.Parse(secondsEnv);
            string coresEnv = Environment.GetEnvironmentVariable("ETERNITY_VIRTUAL_CORES");
            if (!string.IsNullOrWhiteSpace(coresEnv)) number_virtual_cores = int.Parse(coresEnv);

            string breakIndexesDisplay = Environment.GetEnvironmentVariable("ETERNITY_BREAK_INDEXES") ?? "(default 201..239)";
            Console.WriteLine("=== RUN {0}: break_indexes={1}, node_cap={2}, tune_seconds={3} ===",
                runLabel, breakIndexesDisplay,
                node_cap, tuneSeconds.HasValue ? tuneSeconds.Value.ToString() : "(unbounded)");

            Stopwatch runClock = Stopwatch.StartNew();

            // Per-attempt stats, derived from SolvePuzzle()'s own returned
            // histogram rather than changing its signature: the highest index
            // with a nonzero count IS that attempt's max depth (every visited
            // index gets incremented, backtracking past it doesn't erase the
            // count left at the peak), and the sum across all 257 entries IS
            // that attempt's total node count (both incremented once per loop
            // iteration, same iteration, in SolvePuzzle()). Zero risk to the
            // actual search logic -- this only reads what it already computes.
            ConcurrentBag<int> attemptMaxDepths = new ConcurrentBag<int>();
            ConcurrentBag<long> attemptNodeCounts = new ConcurrentBag<long>();
            long totalAttempts = 0;

            while (true) // Solve for Eternity.
            {
                if (tuneSeconds.HasValue && runClock.Elapsed.TotalSeconds >= tuneSeconds.Value)
                    break;

                Prepare_Pieces_And_Heuristics();

                Console.WriteLine("Solving... ({0:F0}s elapsed)", runClock.Elapsed.TotalSeconds);

                ConcurrentDictionary<int, long> index_counts = new ConcurrentDictionary<int, long>();

                // This only runs number_vcpu-1 threads; we need to save one for the us
                var result = Parallel.For(1, number_virtual_cores, (i, state) =>
                {
                    for (int x = 0; x < 5; x++)
                    {
                        Stopwatch stopwatch = new Stopwatch();
                        stopwatch.Start();

                        long[] solve_indexes = SolvePuzzle();

                        int attemptMax = 0;
                        long attemptNodes = 0;
                        for (int j = 0; j < 257; j++)
                        {
                            attemptNodes += solve_indexes[j];
                            if (solve_indexes[j] > 0) attemptMax = j;
                        }
                        attemptMaxDepths.Add(attemptMax);
                        attemptNodeCounts.Add(attemptNodes);
                        Interlocked.Increment(ref totalAttempts);

                        for (int j = 0; j < 257; j++)
                            index_counts.AddOrUpdate(j, solve_indexes[j], (id, count) => count + solve_indexes[j]);

                        stopwatch.Stop();
                    }
                });

                if (tuneSeconds.HasValue) continue; // bounded tuning runs: skip per-generation output, final summary covers it

                // Indefinite (real-attempt) runs: report progress every generation
                // instead of only at an end that never arrives. Cumulative across
                // the whole run, not just this generation -- 12 bytes/attempt in
                // the two bags, so even days of runtime stays negligible; no need
                // to discard history for a number this small.
                var genSorted = attemptMaxDepths.OrderByDescending(d => d).ToList();
                long genNodes = attemptNodeCounts.Sum();
                Console.WriteLine("=== GEN {0}: {1} attempts this run, {2} total nodes, {3:F0}s elapsed, best_depth={4} ===",
                    runLabel, totalAttempts, genNodes, runClock.Elapsed.TotalSeconds, genSorted.Count > 0 ? genSorted[0] : 0);
                foreach (int threshold in new[] { 245, 248, 249, 250, 251, 252, 253, 254, 255, 256 })
                {
                    long reachedCount = genSorted.Count(d => d >= threshold);
                    if (reachedCount > 0)
                        Console.WriteLine("GEN_REACH {0}\t{1}\t{2:F4}", threshold, reachedCount, (double)reachedCount / Math.Max(totalAttempts, 1));
                }
            }

            if (tuneSeconds.HasValue)
            {
                var sorted = attemptMaxDepths.OrderByDescending(d => d).ToList();
                long totalNodes = attemptNodeCounts.Sum();
                double elapsed = runClock.Elapsed.TotalSeconds;

                Console.WriteLine();
                Console.WriteLine("=== SUMMARY {0}: {1} attempts, {2} total nodes, {3:F1}s, {4:F0} nodes/sec ===",
                    runLabel, totalAttempts, totalNodes, elapsed, totalNodes / Math.Max(elapsed, 0.001));

                foreach (int threshold in new[] { 200, 220, 240, 245, 248, 249, 250, 251, 252, 253, 254, 255, 256 })
                {
                    long reachedCount = sorted.Count(d => d >= threshold);
                    Console.WriteLine("REACH {0}\t{1}\t{2:F4}", threshold, reachedCount, (double)reachedCount / Math.Max(totalAttempts, 1));
                }

                Console.WriteLine("BEST_DEPTH\t{0}", sorted.Count > 0 ? sorted[0] : 0);
                Console.WriteLine("=== END SUMMARY {0} ===", runLabel);
            }
        }

        static unsafe long[] SolvePuzzle()
        {
            bool* piece_used = stackalloc bool[257];
            byte* cumulative_heuristic_side_count = stackalloc byte[256];
            byte* piece_index_to_try_next = stackalloc byte[256];
            byte* cumulative_breaks = stackalloc byte[256];
            long[] solve_index_counts = new long[257];
            RotatedPiece* board = stackalloc RotatedPiece[256];

            Random rand = new Random();

            var bottom_sides = new RotatedPiece[529][];
            foreach (var m in bottom_side_pieces_rotated)
                bottom_sides[m.Key] = m.Value.OrderByDescending(x => (x.RotatedPiece.Heuristic_Side_Count > 0 ? 100 : 0) + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            board[0] = corners[0].ToList().OrderBy(x => rand.Next(1, 1000)).First(); // Get rid of pieces 1 or 2 first
            piece_used[board[0].PieceNumber] = true;
            cumulative_breaks[0] = 0;
            cumulative_heuristic_side_count[0] = board[0].Heuristic_Side_Count;

            int solve_index = 1; // this goes from 0....255; we've solved #0 already, so start at #1.
            int max_solve_index = solve_index;
            long node_count = 0;

            while (true)
            {
                node_count++;

                solve_index_counts[solve_index] = solve_index_counts[solve_index] + 1;

                if (solve_index > max_solve_index)
                {
                    max_solve_index = solve_index;

                    // Upstream saves only at >= 252. Local value was 190, which wrote
                    // a file for every new max depth from 190 up and produced 389,856
                    // boards -- 376,919 of them below 248 and read by nothing, since
                    // the conflict tracker's floor is 248. They sat in a
                    // OneDrive-redirected folder, so all of it synced to the cloud and
                    // took the quota to 80% full.
                    //
                    // Matched to the tracker's MIN_DEPTH. A 24-board sample of the
                    // 245-247 band measured 16-21 conflicts (means 18.6-19.5) against
                    // a record of 12 from a 251-piece board, so nothing down there was
                    // worth the disk: shallower boards leave more holes, and hole-fill
                    // cost is what dominates the final count. Keep these two numbers
                    // in step -- saving below the tracker's floor writes files that
                    // nothing will ever read.
                    //
                    // 2026-09-02: temporarily dropped to 219, matching Eternity2_GPU's
                    // HARVEST_MIN_DEPTH -- the 5-clue pin fix just went in and this run
                    // hasn't reached 248 yet, so lowering it gets an early completed
                    // board out for inspection instead of waiting on the first record
                    // past 248. Raise back once satisfied the fix is behaving.
                    if (solve_index >= 219)
                    {
                        RotatedPiece[] board_to_save = new RotatedPiece[256];

                        for (int i = 0; i < 256; i++)
                            board_to_save[i] = board[i]; // convert to managed just in case

                        Util.Save_Board(board_to_save, (ushort)solve_index);

                        if (solve_index >= 256)
                            return solve_index_counts;
                    }
                }

                if (node_count > node_cap)
                {
                    return solve_index_counts;
                }

                byte row = board_search_sequence[solve_index].Row;
                byte col = board_search_sequence[solve_index].Column;

                if (board[row * 16 + col].PieceNumber > 0)
                {
                    piece_used[board[row * 16 + col].PieceNumber] = false;
                    board[row * 16 + col].PieceNumber = 0;
                }

                RotatedPiece[] piece_candidates;

                if (row == 0)
                {
                    if (col < 15)
                        piece_candidates = bottom_sides[board[row * 16 + (col - 1)].RightSide * 23 + 0];
                    else
                    {
                        piece_candidates = corners[board[row * 16 + (col - 1)].RightSide * 23 + 0];
                    }
                }
                else
                {
                    var leftSide = (col == 0) ? 0 : board[row * 16 + (col - 1)].RightSide;
                    piece_candidates = master_piece_lookup[row * 16 + col][leftSide * 23 + board[(row - 1) * 16 + col].TopSide];
                }

                bool found_piece = false;
                if (piece_candidates != null)
                {
                    byte breaks_this_turn = (byte)(break_array[solve_index] - cumulative_breaks[solve_index - 1]);
                    int try_index = piece_index_to_try_next[solve_index];

                    int pieceCandidateLength = piece_candidates.Length;
                    for (int i = try_index; i < pieceCandidateLength; i++)
                    {
                        if (piece_candidates[i].Break_Count > breaks_this_turn)
                            break;

                        if (!piece_used[piece_candidates[i].PieceNumber])
                        {
                            if (solve_index <= max_heuristic_index)
                            {
                                if ((cumulative_heuristic_side_count[solve_index - 1] + piece_candidates[i].Heuristic_Side_Count) < heuristic_array[solve_index])
                                    break;
                            }

                            found_piece = true;

                            var piece = piece_candidates[i];

                            board[row * 16 + col] = piece;
                            piece_used[piece.PieceNumber] = true;

                            cumulative_breaks[solve_index] = (byte)(cumulative_breaks[solve_index - 1] + piece.Break_Count);
                            cumulative_heuristic_side_count[solve_index] = (byte)(cumulative_heuristic_side_count[solve_index - 1] + piece.Heuristic_Side_Count);

                            piece_index_to_try_next[solve_index] = (byte)(i + 1);
                            solve_index++;
                            break;
                        }
                    }
                }

                if (!found_piece)
                {
                    piece_index_to_try_next[solve_index] = 0;
                    solve_index--;
                }
            }
        }

        static void Prepare_Pieces_And_Heuristics()
        {
            var board_pieces = Util.Get_Pieces();
            var corner_pieces = board_pieces.Where(x => x.PieceType() == 2).ToList();
            var side_pieces = board_pieces.Where(x => x.PieceType() == 1).ToList();
            var middle_pieces = board_pieces.Where(x => x.PieceType() == 0)
                .Where(x => x.PieceNumber != 139 && x.PieceNumber != 208 && x.PieceNumber != 255 && x.PieceNumber != 181 && x.PieceNumber != 249)
                .ToList(); // exclude the 5 official clue pieces
            var start_piece = board_pieces.Where(x => x.PieceNumber == 139).ToList();
            var hint208_piece = board_pieces.Where(x => x.PieceNumber == 208).ToList();
            var hint255_piece = board_pieces.Where(x => x.PieceNumber == 255).ToList();
            var hint181_piece = board_pieces.Where(x => x.PieceNumber == 181).ToList();
            var hint249_piece = board_pieces.Where(x => x.PieceNumber == 249).ToList();

            // corners
            var corner_pieces_rotated = corner_pieces.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());

            // sides
            var sides_without_breaks = side_pieces.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x);
            var sides_with_breaks = side_pieces.Select(x => Util.Get_Rotated_Pieces(x, true)).SelectMany(x => x);
            bottom_side_pieces_rotated = sides_without_breaks.Where(x => x.RotatedPiece.Rotations == 0).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var left_side_pieces_rotated = sides_without_breaks.Where(x => x.RotatedPiece.Rotations == 1).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var right_side_pieces_with_breaks_rotated = sides_with_breaks.Where(x => x.RotatedPiece.Rotations == 3).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var right_side_pieces_without_breaks_rotated = sides_without_breaks.Where(x => x.RotatedPiece.Rotations == 3).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var top_side_pieces_rotated = sides_with_breaks.Where(x => x.RotatedPiece.Rotations == 2).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());

            // middles
            var middle_pieces_rotated_with_breaks = middle_pieces.Select(x => Util.Get_Rotated_Pieces(x, true)).SelectMany(x => x).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var middle_pieces_rotated_without_breaks = middle_pieces.Select(x => Util.Get_Rotated_Pieces(x, false)).SelectMany(x => x).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var south_start_piece_rotated = middle_pieces.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.TopSide == 6).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var west_start_piece_rotated = middle_pieces.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.RightSide == 11).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var start_piece_rotated = start_piece.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.Rotations == 2).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var hint208_piece_rotated = hint208_piece.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.Rotations == 2).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var hint255_piece_rotated = hint255_piece.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.Rotations == 2).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var hint181_piece_rotated = hint181_piece.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.Rotations == 2).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());
            var hint249_piece_rotated = hint249_piece.Select(x => Util.Get_Rotated_Pieces(x)).SelectMany(x => x).Where(x => x.RotatedPiece.Rotations == 3).GroupBy(x => x.LeftBottom).ToDictionary(x => x.Key, y => y.ToList());

            Random rand = new Random();

            corners = new RotatedPiece[529][];
            foreach (var m in corner_pieces_rotated)
                corners[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            left_sides = new RotatedPiece[529][];
            foreach (var m in left_side_pieces_rotated)
                left_sides[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            top_sides = new RotatedPiece[529][];
            foreach (var m in top_side_pieces_rotated)
                top_sides[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            right_sides_with_breaks = new RotatedPiece[529][];
            foreach (var m in right_side_pieces_with_breaks_rotated)
                right_sides_with_breaks[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            right_sides_without_breaks = new RotatedPiece[529][];
            foreach (var m in right_side_pieces_without_breaks_rotated)
                right_sides_without_breaks[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            middles_with_break = new RotatedPiece[529][];
            foreach (var m in middle_pieces_rotated_with_breaks)
                middles_with_break[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            middles_no_break = new RotatedPiece[529][];
            foreach (var m in middle_pieces_rotated_without_breaks)
                middles_no_break[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            south_start = new RotatedPiece[529][];
            foreach (var m in south_start_piece_rotated)
                south_start[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            west_start = new RotatedPiece[529][];
            foreach (var m in west_start_piece_rotated)
                west_start[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            start = new RotatedPiece[529][];
            foreach (var m in start_piece_rotated)
                start[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            hint208 = new RotatedPiece[529][];
            foreach (var m in hint208_piece_rotated)
                hint208[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            hint255 = new RotatedPiece[529][];
            foreach (var m in hint255_piece_rotated)
                hint255[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            hint181 = new RotatedPiece[529][];
            foreach (var m in hint181_piece_rotated)
                hint181[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            hint249 = new RotatedPiece[529][];
            foreach (var m in hint249_piece_rotated)
                hint249[m.Key] = m.Value.OrderByDescending(x => x.Score + rand.Next(0, 99)).Select(x => x.RotatedPiece).ToArray();

            board_search_sequence = Util.Get_Board_Order();
            break_array = Util.Get_Break_Array();
            master_piece_lookup = new RotatedPiece[256][][];
            for (int i = 0; i < 256; i++)
            {
                int row = board_search_sequence[i].Row;
                int col = board_search_sequence[i].Column;

                if (row == 15)
                {
                    if ((col == 15) || (col == 0))
                        master_piece_lookup[row * 16 + col] = corners;
                    else
                        master_piece_lookup[row * 16 + col] = top_sides;
                }
                else if (row == 0)
                {
                    // Don't populate the master lookup table since we randomize every time.
                }
                else
                {
                    if (col == 15)
                    {
                        if (i < Util.First_Break_Index())
                            master_piece_lookup[row * 16 + col] = right_sides_without_breaks;
                        else
                            master_piece_lookup[row * 16 + col] = right_sides_with_breaks;
                    }
                    else if (col == 0)
                        master_piece_lookup[row * 16 + col] = left_sides;
                    else
                    {
                        if (row == 7)
                        {
                            if (col == 7)
                                master_piece_lookup[row * 16 + col] = start;
                            else if (col == 6)
                                master_piece_lookup[row * 16 + col] = west_start;
                            else
                            {
                                if (i < Util.First_Break_Index())
                                    master_piece_lookup[row * 16 + col] = middles_no_break;
                                else
                                    master_piece_lookup[row * 16 + col] = middles_with_break;
                            }
                        }
                        else if (row == 6)
                        {
                            if (col == 7)
                                master_piece_lookup[row * 16 + col] = south_start;
                            else
                            {
                                if (i < Util.First_Break_Index())
                                    master_piece_lookup[row * 16 + col] = middles_no_break;
                                else
                                    master_piece_lookup[row * 16 + col] = middles_with_break;
                            }
                        }
                        else if (row == 2 && col == 2)
                            master_piece_lookup[row * 16 + col] = hint181;
                        else if (row == 2 && col == 13)
                            master_piece_lookup[row * 16 + col] = hint249;
                        else if (row == 13 && col == 2)
                            master_piece_lookup[row * 16 + col] = hint208;
                        else if (row == 13 && col == 13)
                            master_piece_lookup[row * 16 + col] = hint255;
                        else
                        {
                            if (i < Util.First_Break_Index())
                                master_piece_lookup[row * 16 + col] = middles_no_break;
                            else
                                master_piece_lookup[row * 16 + col] = middles_with_break;
                        }
                    }
                }
            }

            heuristic_array = new int[256];
            for (int i = 0; i < 256; i++)
            {
                if (i <= 16)
                    heuristic_array[i] = 0;
                else if (i <= 26)
                    heuristic_array[i] = (int)(((float)i - 16) * (float)2.8);
                else if (i <= 56)
                    heuristic_array[i] = (int)((((float)i - 26) * (float)1.43333) + 28);
                else if (i <= 76)
                    heuristic_array[i] = (int)(((((float)i - 56) * (float)0.9)) + 71);
                else if (i <= 102)
                    heuristic_array[i] = (int)(((((float)i - 76) * (float)0.6538)) + 89);
                else if (i <= max_heuristic_index)
                    heuristic_array[i] = (int)(((((float)i - 102) / 4.4615)) + 106);
            }
        }

        static RotatedPiece[][] corners;
        static RotatedPiece[][] left_sides;
        static RotatedPiece[][] right_sides_with_breaks;
        static RotatedPiece[][] right_sides_without_breaks;
        static RotatedPiece[][] top_sides;
        static RotatedPiece[][] middles_with_break;
        static RotatedPiece[][] middles_no_break;
        static RotatedPiece[][] south_start;
        static RotatedPiece[][] west_start;
        static RotatedPiece[][] start;
        // The 4 non-center official Eternity II clues (piece 139/start is the 5th). Position and
        // rotation independently re-derived and cross-checked three ways -- see the javadoc on
        // BlackwoodSolver.HINT_PINS in the Java ports for the full derivation; identical values
        // apply here unchanged since Get_Rotated_Pieces uses the same (T,R,B,L) rotation mapping.
        static RotatedPiece[][] hint208;
        static RotatedPiece[][] hint255;
        static RotatedPiece[][] hint181;
        static RotatedPiece[][] hint249;
        static Dictionary<ushort, List<RotatedPieceWithLeftBottom>> bottom_side_pieces_rotated;
        static RotatedPiece[][][] master_piece_lookup;

        static SearchIndex[] board_search_sequence;
        static byte[] break_array;
        static int[] heuristic_array;
        const int max_heuristic_index = 160;
    }
}
