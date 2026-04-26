"""aura_bench — public benchmark + ablation harness.

Every score has a command, a metric, a log, a baseline, and an ablation.
Pre-registration is required: each test ships with a hypothesis, metric,
threshold, and number of trials in the test file's docstring before the
test is run.
"""
