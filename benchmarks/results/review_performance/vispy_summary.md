# Comparative measurements

P95 values are medians of per-run P95s. CPU is percent of one core. Draw Hz is software draw completion, not
presentation FPS.

| Track     | Target | Scenario   | Mode       | Qt P95 ms | VisPy P95 ms | Reduction | Qt / VisPy CPU % | Evidence             |
| --------- | -----: | ---------- | ---------- | --------: | -----------: | --------: | ---------------: | -------------------- |
| universal |     60 | follow_10  | rendering  |     60.56 |        31.51 |     48.0% |     100.5 / 94.8 | 3 x 30 s             |
| universal |    120 | follow_10  | integrated |     55.92 |        28.13 |     49.7% |     103.0 / 98.7 | 1 x 3 s; exploratory |
| universal |     60 | follow_120 | rendering  |    104.55 |        39.89 |     61.8% |      80.2 / 80.4 | 2 x 30 s; draw gaps  |
| universal |     60 | follow_2.5 | rendering  |     47.30 |        28.05 |     40.7% |     100.7 / 92.5 | 3 x 30 s             |
| universal |     60 | follow_40  | rendering  |     81.56 |        36.77 |     54.9% |     100.4 / 93.8 | 3 x 30 s             |
| universal |    120 | seek       | integrated |     72.34 |        43.52 |     39.8% |    110.6 / 112.9 | 1 x 3 s; exploratory |
