# Submitting a router

1. Write a factory `make(ctx) -> Router` (see `examples/my_router.py` and `capbench/api.py`). `ctx` is a `capbench.run.RouterContext`. Name the router with a `name` attribute. Prefix it `A:` if it is a Track A router (its own predictor, no `ctx.P` or `Query.row`).
2. Run the full suite (under 25 min with the four scenarios in parallel via `--scenario`):

   ```bash
   python -m capbench.run --suite v1 --router my_pkg.my_router:make --only-custom --tag v1_myrouter
   python -m capbench.score outputs/v1_myrouter.csv --markdown
   ```

3. Open a pull request that adds:
   - `submissions/v1/<router-name>/results.csv`: the CSV from step 2, unedited.
   - `submissions/v1/<router-name>/README.md`: the track, a one-paragraph method description, a link to code (or the code itself under `submissions/`), and confirmation that you followed the rules in `docs/PROTOCOL.md`.

   A maintainer re-scores the CSV, spot-checks a few cells by rerunning your code, and adds the router to the leaderboard in `README.md`.

Results from an older suite version stay on that version's leaderboard. A new suite version starts its own.
