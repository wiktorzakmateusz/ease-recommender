# Benchmarking

To check the gain obtained by using the package, use $benchmark.py$ code:

```bash
python benchmark.py --num_threads=10 --repeats=5
```

expected outcome:

```
Generating synthetic sparse dataset...
Users: 100000 | Items: 5000 | Density: 0.5%
--------------------------------------------------
Starting Benchmark (5 iterations, 10 threads)...
  Running iteration 1/5...
  Running iteration 2/5...
  Running iteration 3/5...
  Running iteration 4/5...
  Running iteration 5/5...
--------------------------------------------------
RESULTS (Averaged over 5 runs):

Fit (Training) Times
Python:       1.8773s +/- 0.0224s
C-Optimized:  0.4804s +/- 0.0039s
Speedup:      3.91x faster

Inference Times
Python:       0.017600s +/- 0.000705s
C-Optimized:  0.000030s +/- 0.000001s
Speedup:      581.66x faster
--------------------------------------------------
```