# olmo-hands-on

Learning Ai2's Olmo training stack by running it outside Ai2, on 4× RTX 4090: setup notes, fixes, and small analyses for [open-instruct](https://github.com/allenai/open-instruct) issues.

Independent work by [Qirun Li](https://github.com/bethehand). Not affiliated with Ai2.

## Contributions

| Date | Where | What | Status |
|---|---|---|---|
| 2026-10-17 | open-instruct | The desktop RTX 4090 is missing from `GPU_SPECS`, so `grpo_fast.py` stops at startup with `Unknown device name`. One-line fix plus a test. | Reproduced on the machine; fix and test ready; PR coming |

## Layout

| Path | What's there |
|---|---|
| [`setup/`](setup/) | Running open-instruct's GRPO on 4× RTX 4090 outside Ai2's cluster: steps, scripts, and every snag I hit |
| `analyses/` | Small analyses for specific open-instruct issues (added as they happen) |

## Related

- [olmo-rlvr-side-effects](https://github.com/bethehand/olmo-rlvr-side-effects): my research project on Ai2's RLVR recipe. Which training data changes what, outside the training domain?
