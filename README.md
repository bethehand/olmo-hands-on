# olmo-hands-on

Learning Ai2's Olmo training stack by running it outside Ai2, on 4× RTX 4090: setup notes, fixes, and an index of my contributions to [open-instruct](https://github.com/allenai/open-instruct).

Independent work by [Qirun Li](https://github.com/bethehand). Not affiliated with Ai2.

## Contributions

| Date | Where | What | Status |
|---|---|---|---|
| 2026-10-17 | open-instruct [#2018](https://github.com/allenai/open-instruct/pull/2018) | The desktop RTX 4090 is missing from `GPU_SPECS`, so `grpo_fast.py` stops at startup with `Unknown device name`. One entry plus a test. | Open, waiting for review |
| 2026-10-18 | open-instruct [#1964](https://github.com/allenai/open-instruct/issues/1964#issuecomment-6247442515) (comment) | Safety baselines and run-to-run noise for the Olmo 3 7B Think stages (XSTest, WildJailbreak-benign, WildGuardTest; 3 runs each, within 1 point of Table 52). Also: safety-eval reruns can come out byte-identical, because no seed is passed. Code and results: [`think7b-safety`](https://github.com/bethehand/olmo-rlvr-side-effects/tree/main/experiments/2026-10-18_think7b-safety) | Posted |
| 2026-10-19 | open-instruct [#2002](https://github.com/allenai/open-instruct/issues/2002#issuecomment-6256732503) (comment) | Answer length across the SFT, DPO and final stages of Olmo 3.1 32B Instruct, Olmo 3 / 3.1 32B Think and Olmo 3 7B Think: DPO adds most of the length; 3.1's longer RL adds some, hidden by its new default system prompt. Code and results: [`length`](https://github.com/bethehand/olmo-rlvr-side-effects/tree/main/experiments/2026-10-18_length) | Posted |

## Layout

| Path | What's there |
|---|---|
| [`setup/`](setup/) | Running open-instruct's GRPO on 4× RTX 4090 outside Ai2's cluster: steps, scripts, and every snag I hit |

Analyses for open-instruct issues live in [olmo-rlvr-side-effects](https://github.com/bethehand/olmo-rlvr-side-effects), one dated folder per experiment. This repo keeps how to run the tools and the list of contributions.

## Related

- [olmo-rlvr-side-effects](https://github.com/bethehand/olmo-rlvr-side-effects): my research project on Ai2's RLVR recipe. Which training data changes what, outside the training domain?
