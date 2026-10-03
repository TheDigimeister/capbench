# Suite v1 baseline results

These results cover 10 baselines on 4 scenarios × 4 load levels × 2 budgets × 5 seeds, for 160 cells per router (`outputs/v1_s*.csv`, plotted in `outputs/v1.png`). **Caveat:** a live validation (see [CALIBRATION.md](CALIBRATION.md)) found that the simulator overstates self-hosted capacity by about 1.8×. These results are provisional: at a given ρ, real self-hosted queues would be longer, which mostly penalises routers that rely on self-hosted capacity.

## CapScore by scenario

| Router | H100 Poisson | H100 BurstGPT | Spark Poisson | Spark BurstGPT | CapScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| *fluid ceiling (reference)* | 0.889 | 0.887 | 0.882 | 0.883 | 0.885 |
| `batch_lp@2s` | 0.716 | 0.681 | 0.709 | 0.661 | **0.692** |
| `capacity_greedy` | 0.712 | 0.670 | 0.707 | 0.648 | 0.684 |
| `A:knn_capacity_greedy` | 0.671 | 0.624 | 0.655 | 0.597 | 0.637 |
| `static_qc_fallback` | 0.631 | 0.537 | 0.614 | 0.517 | 0.575 |
| `static_qc` | 0.604 | 0.448 | 0.595 | 0.455 | 0.526 |
| `shortest_queue` | 0.424 | 0.312 | 0.549 | 0.384 | 0.417 |

## Findings

1. **Load awareness is worth about 0.11 CapScore.** `capacity_greedy` uses the same predictor and price weight as `static_qc_fallback`, but checks free slots, queue room, rate-limit headroom and budget before routing. It beats the fallback router in all 160 cells (smallest margin +0.012). `batch_lp@2s`, which re-solves a small assignment LP every 2 s, adds about 0.008 more and loses to greedy in 33 of 160 cells, each time by at most 0.010.

2. **Bursty arrivals cost every router, and they cost load-blind routers most.** From Poisson to BurstGPT, `batch_lp` drops 0.04 but `static_qc_fallback` drops 0.10 and `static_qc` drops 0.15. The diagnostics at budget 0.1 show the mechanism:

   | BurstGPT, budget 0.1 | SLO attainment | 429 refusals / query | budget refusals / query | queue-full refusals / query |
   | --- | ---: | ---: | ---: | ---: |
   | `batch_lp@2s` | 0.90 | 0.01 | 0.00 | 0.00 |
   | `capacity_greedy` | 0.88 | 0.02 | 0.00 | 0.16 |
   | `static_qc_fallback` | 0.65 | 0.34 | 0.42 | 0.09 |
   | `static_qc` | 0.56 | 0.58 | 0.30 | 0.08 |

   A burst sends many queries to the same preferred API model at once, which drains its per-minute bucket (429s). The fallback chain then retries with the next-best model by quality minus price, usually another paid API, with no view of headroom. That spends the trailing-window budget early in a burst (its mean spend is the highest, $138/h against $127/h for the load-aware routers), and the rest of the window is refused on budget. The load-aware routers instead spill to DeepSeek-V3, which has no published rate limit (32–35% of BurstGPT traffic against 20–23% under Poisson), and to self-hosted models while slots are free. `capacity_greedy`'s residual loss is self-hosted queue overflow (0.16 queue-full refusals per query), which `batch_lp`'s look-ahead avoids.

3. **A wide gap remains.** The best baseline is 0.17 (Poisson) to 0.21 (BurstGPT) below the fluid ceiling. Part of that is the predictor: with no capacity limits at all, routing by the Track B predictor scores 0.814 against the oracle's 0.928. The rest is allocation under bursts, which the ceiling relaxes away. A router that forecasts bursts or reserves budget across a window has room to win.

4. **Track A vs Track B.** The TF-IDF kNN example (`A:knn_capacity_greedy`) trails the same allocator with the shared embedding predictor by 0.048. That gap is roughly what a better text predictor could recover.

5. **Open question.** On BurstGPT at budget 0.1, the load-aware routers score *higher* at higher load (`batch_lp` goes from 0.616 at ρ = 0.3 to 0.675 at ρ = 1.3). The budget scales with the mean arrival rate, so this is not simply more money at higher load. The cause is not yet established; it persists with fixed-duration runs, and the next thing to check is how burst intensity within a 10-minute budget window changes with the rescaled trace rate.

## Earlier results

Runs made before suite v1 used a train/test split that changed with each seed, and per-seed predictor caches that were not keyed to the split. They are not comparable and are not published.
