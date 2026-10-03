# Baseline results

## Suite v2 (current)

These results cover 10 baselines on the DGX Spark profile `spark_v2`, which is validated against live vLLM ([CALIBRATION.md](CALIBRATION.md)). The grid is 2 arrival processes × 4 load levels × 2 budgets × 5 seeds, for 80 cells per router (`outputs/v2_s*_seed*.csv`, plotted in `outputs/v2.png`).

| Router | Poisson | BurstGPT | CapScore |
| --- | ---: | ---: | ---: |
| *fluid ceiling (reference)* | 0.879 | 0.880 | 0.879 |
| `batch_lp@2s` | 0.701 | 0.654 | **0.678** |
| `capacity_greedy` | 0.699 | 0.641 | 0.670 |
| `A:knn_capacity_greedy` | 0.650 | 0.591 | 0.621 |
| `static_qc_fallback` | 0.611 | 0.513 | 0.562 |
| `static_qc` | 0.593 | 0.452 | 0.522 |
| `shortest_queue` | 0.552 | 0.386 | 0.469 |

### Findings

1. **Load awareness is worth about 0.11 CapScore.** `capacity_greedy` uses the same predictor and price weight as `static_qc_fallback`, but checks free slots, queue room, rate-limit headroom and budget before routing. It beats the fallback router in all 80 cells (smallest margin +0.005, mean +0.108). `batch_lp@2s` re-solves a small assignment LP every 2 s. It adds about 0.007 more on average and loses to greedy in 18 of 80 cells, each time by at most 0.016.

2. **Bursty arrivals cost every router, and they cost load-blind routers most.** From Poisson to BurstGPT, `batch_lp` drops 0.05, `static_qc_fallback` 0.10 and `static_qc` 0.14. The diagnostics at budget 0.1 show the mechanism:

   | BurstGPT, budget 0.1 | SLO attainment | 429 refusals / query | budget refusals / query | queue-full refusals / query | API spend |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | `batch_lp@2s` | 0.85 | 0.01 | 0.00 | 0.00 | $105/h |
   | `capacity_greedy` | 0.82 | 0.02 | 0.00 | 0.20 | $104/h |
   | `static_qc_fallback` | 0.59 | 0.27 | 0.50 | 0.08 | $114/h |
   | `static_qc` | 0.54 | 0.48 | 0.37 | 0.07 | $106/h |

   A burst sends many queries to the same preferred API model at once, which drains its per-minute bucket (429s). The fallback chain then retries with the next-best model by quality minus price, usually another paid API, with no view of headroom. That spends the trailing-window budget early in the burst, and the rest of the window is refused on budget (0.50 refusals per query). The load-aware routers instead spill to DeepSeek-V3, which has no published rate limit (32–34% of BurstGPT traffic against 19–21% under Poisson). `capacity_greedy`'s residual loss is self-hosted queue overflow (0.20 queue-full refusals per query), which `batch_lp`'s look-ahead avoids.

3. **The self-hosted tier is marginal in this pool.** The load-aware routers send only 1.6–2.3% of queries to the three self-hosted 8–9B models. Those models are much less accurate than the API models, and their capacity is a small part of total capacity. This is why correcting the self-hosted simulator (v1 → v2, about 1.8× less capacity) moved every top router by at most 0.007 and left the ranking unchanged. The benchmark currently measures mainly how well routers handle API rate limits and a windowed budget. A pool with stronger or more self-hosted models would test self-hosted queueing harder.

4. **A wide gap remains.** The best baseline is 0.18 (Poisson) to 0.23 (BurstGPT) below the fluid ceiling. Part of that is the predictor: with no capacity limits at all, routing by the Track B predictor scores 0.813 against the oracle's 0.928. The rest is allocation under bursts, which the ceiling relaxes away. A router that forecasts bursts or reserves budget across a window has room to win.

5. **Track A vs Track B.** The TF-IDF kNN example (`A:knn_capacity_greedy`) trails the same allocator with the shared embedding predictor by 0.050.

6. **Open question.** On BurstGPT at budget 0.1, the load-aware routers score *higher* at higher load (`batch_lp` goes from 0.577 at ρ = 0.3 to 0.658 at ρ = 1.3). The budget scales with the mean arrival rate, so this is not simply more money at higher load. The cause is not yet established. The next thing to check is how burst intensity within a 10-minute budget window changes with the rescaled trace rate.

## Suite v1 (provisional)

Suite v1 (`outputs/v1_s*.csv`, `outputs/v1.png`) adds an H100 profile and uses v1 self-hosted profiles, which are not validated. Its CapScores are batch_lp 0.692, capacity_greedy 0.684, kNN 0.637, static_qc_fallback 0.575 and static_qc 0.526. Its Spark scenarios match suite v2's to within 0.007 for every router above, for the reason given in finding 3.

Runs made before suite v1 used a train/test split that changed with each seed, and predictor caches that were not keyed to the split. They are not comparable and are not published.
