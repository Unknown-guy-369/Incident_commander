# Incident Commander — Hackathon Project Review (Revised)

Reviewed against the four hackathon PDFs (Themes & Judging, What-Judges-Look-For,
Resources, Reward-engineering FAQ), the latest `colab/incident_commander_training.ipynb`,
the `trained_halfway (1).ipynb` upload, and `HACKATHON_MEMORY.md`.

> **Correction to the previous version of this file.** The first pass of this
> review claimed `models.py`, `client.py`, `training.py`, `inference.py`,
> `app.py`, `server/app.py`, `Dockerfile`, `server/Dockerfile`,
> `pyproject.toml`, and `colab/incident_commander_training.ipynb` were missing.
> They are not. The earlier `Glob` listing was stale and got truncated by the
> `venv/` tree. A direct `ls` confirms all of those files are in the repo,
> created today at ~17:04. The "B1 — missing files" blocker from the previous
> draft is **withdrawn**. The rest of the analysis stands and a few new
> findings replace it.

---

## File inventory (confirmed present)

```
models.py                                       ✅
client.py                                       ✅
simulator.py                                    ✅
rewards.py                                      ✅
training.py                                     ✅
inference.py                                    ✅
inference_openrouter.py                         ✅
app.py                                          ✅ (root bootstrap)
Dockerfile                                      ✅ (root, near-duplicate of server/)
__init__.py                                     ✅
openenv.yaml                                    ✅
pyproject.toml                                  ✅
requirements.txt                                ✅
README.md, DEPLOY.md, HACKATHON_MEMORY.md       ✅
server/app.py                                   ✅
server/Dockerfile                               ✅
server/incident_commander_environment.py        ✅
server/__init__.py                              ✅
server/requirements.txt                         ✅
colab/incident_commander_training.ipynb         ✅
tests/{smoke_test,test_env,test_training}.py    ✅
project_report_complete.md                      ❌ referenced in HACKATHON_MEMORY but absent
```

---

## Status against hackathon minimum requirements

| Requirement | Status |
|---|---|
| Theme alignment (3.1 Professional Tasks) | ✅ excellent fit |
| OpenEnv `Environment` base class | ✅ |
| Gym-style `reset` / `step` / `state` | ✅ |
| Valid `openenv.yaml` | ✅ |
| No reserved tool names reused as actions | ✅ |
| Client / server separation | ⚠️ `client.py` has a bare `from models import ...` that breaks under packaged install (see new B5) |
| Hosted on HF Spaces | ✅ live at `abishek-priyan-369-incident-commander.hf.space` (per `HACKATHON_MEMORY`) |
| Working TRL / Unsloth training script in Colab | ⚠️ runs, but the rollout shape neutralises the env (see B2) |
| Real loss / reward plots from a real run | ⚠️ latest notebook produces real before/after numbers; `trained_halfway (1).ipynb` had hardcoded "before" values |
| README that motivates problem, explains env, shows results | ⚠️ structure good; no embedded plots, no Space URL, no writeup link |
| Mini-blog or < 2 min video | ❌ not yet present in repo |

---

## Real blockers (after re-review)

### B1. Duplicated `try/except` in `server/incident_commander_environment.py`

Lines 22–29 — both branches do exactly the same thing:

```python
try:
    from models import IncidentCommanderAction, IncidentCommanderObservation
    from simulator import Simulator, CORRECT_FIX
    from rewards import compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS
except (ImportError, ModuleNotFoundError):
    from models import IncidentCommanderAction, IncidentCommanderObservation   # same!
    from simulator import Simulator, CORRECT_FIX
    from rewards import compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS
```

The fallback was clearly meant to be the package-relative form. Replace with:

```python
except (ImportError, ModuleNotFoundError):
    from incident_commander.models import IncidentCommanderAction, IncidentCommanderObservation
    from incident_commander.simulator import Simulator, CORRECT_FIX
    from incident_commander.rewards import compute_total_reward, MAX_STEPS, MIN_INVESTIGATION_STEPS
```

`server/app.py` has a similar pattern that's mostly correct, but the
first import line is duplicated identically inside the `except` — same
fix shape applies.

### B2. The reward function shape defeats the multi-step environment

This is the single highest-impact fix and `HACKATHON_MEMORY.md` already
diagnoses the symptom: **"environment_reward_func has been 0.000 throughout
— model hasn't discovered valid action names yet"** and **"Model generates
`<action>` tags correctly but fills them with placeholder text
(`action_name`, `lock-hypothesis`) instead of valid actions"**.

The notebook's `environment_reward_func` (and the standalone
`training.py`'s) does, for every completion:

```python
env = SyncEnvClient(HF_SPACE_URL)
env.reset()
result = env.step(action_type=action_type, target_service=target)  # ONE step
rewards.append(result.get("reward", 0.0))
env.close()
```

Consequences (and these match what `HACKATHON_MEMORY` is showing):

1. The agent never gets to investigate → identify_cause → fix → monitor →
   resolve. The four reward signals you carefully designed
   (`service_recovery`, `root_cause_accuracy`, `action_quality`, `speed`)
   only fire on `resolve` or timeout, both of which require multi-turn
   trajectories. So `env_reward` stays at `0.000` step after step.
2. Any of the fix actions (`restart_pod`, `rollback`, `scale_up`, `hotfix`)
   triggers the `MIN_INVESTIGATION_STEPS = 3` guard immediately —
   investigation count is 0 — and gets a `-0.1` penalty. The model is being
   taught "applying fixes is bad."
3. `reward_speed` requires `post_fix_status == "recovered"`, which is
   unreachable in one step.
4. `reward_action_quality` is heavily penalty-driven; in clamped terms it
   floors at 0. `format_reward_func` is the only signal with non-zero
   variance — that's why the only thing the model improves at is format
   compliance (10 % → 25 %).

The fix is structural: the reward function must run **a multi-turn rollout
inside its own body**. Sketch:

```python
def episode_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 'completion' is the model's whole turn — could be 1 action,
        # could be a planned sequence. Parse out a *plan* of actions.
        plan = parse_action_plan(completion)         # list of (atype, target, hypothesis)
        env = SyncEnvClient(HF_SPACE_URL)
        env.reset()
        last = None
        for atype, target, hyp in plan[:50]:         # MAX_STEPS guard
            last = env.step(action_type=atype, target_service=target,
                            hypothesis=hyp)
            if last.get("done"):
                break
        rewards.append(float(last.get("reward", 0.0)) if last else 0.0)
        env.close()
    return rewards
```

This still has the sequential-HTTP cost, but at least one completion now
corresponds to one episode, and `service_recovery` / `root_cause_accuracy`
can finally fire.

The right-shaped fix uses TRL's OpenEnv integration directly — see
`https://huggingface.co/docs/trl/main/en/openenv`. Then TRL drives the
multi-turn rollout for you and you don't need `SyncEnvClient` at all
(which solves the issue `HACKATHON_MEMORY` already lists as the reason
`SyncEnvClient` exists: "openenv EnvClient is async — doesn't work with
GRPO trainer". TRL's wrapper handles the async/sync bridge.)

If a full multi-turn refactor is out of budget before submission, the
minimum-viable rescue is to **train on prompts that already contain
3+ revealed log/metric blocks**. That makes fix actions reachable and
unblocks the `service_recovery` reward path even with single-step rewards:

```python
def make_pre_investigated_prompt():
    env = SyncEnvClient(HF_SPACE_URL)
    env.reset()
    for svc in ["payment-service", "auth-service", "api-gateway"]:
        env.step(action_type="read_logs", target_service=svc)
    state = env.step(action_type="read_metrics", target_service="payment-service")
    return render_full_state(state["observation"])
```

That is enough to move `env_reward` off zero and give GRPO real signal.

### B3. Server stores per-episode state on the Environment instance

`IncidentCommanderEnvironment` declares `SUPPORTS_CONCURRENT_SESSIONS = True`
but stores `_ctx`, `_actions`, `_locked_hypothesis`, `_post_fix_status`,
`_rollback_count`, `_escalation_count`, `_read_logs_set`,
`_read_metrics_set`, `_last_result`, and `_state` as **instance attributes**
on `self`. `server/app.py` calls `create_app(IncidentCommanderEnvironment, …,
max_concurrent_envs=16)`, but unless `create_app` instantiates a fresh
`Environment` per session under the hood (it doesn't — it shares one), every
concurrent client mutates the same dictionary keys.

This is going to bite the moment you try to do parallel rollouts during
training, and it already explains some of the inconsistency you're seeing
during evaluation when multiple Colab cells share the Space. Fix options:

1. Use OpenEnv's `MCPEnvironment` base class — it gives session-keyed state.
2. Internalise a `dict[session_id, EpisodeContext]` in
   `IncidentCommanderEnvironment` and require `session_id` on every call.
3. Set `SUPPORTS_CONCURRENT_SESSIONS = False`, set `max_concurrent_envs=1`,
   and accept slower rollouts. Honest, but slow.

### B4. Training-run length is not enough to show learning

Per `HACKATHON_MEMORY`:

> Run 1: 35/62 steps before Colab auto-stopped. Model weights LOST.
> Run 2: 20 steps total. env_reward=0.000 across all batches.
> Improvement shown: Format Compliance 10% → 25%.

Two batches × 10 steps = 20 GRPO updates against a model that hasn't even
discovered valid action names yet. That's a smoke test, not a training run.
Criterion 3 (Showing Improvement, 20 %) wants "reward curves, before/after
behavior, comparison against a baseline — anything that proves the agent
learned something." Format-compliance going from 10 % to 25 % is real
learning, but it's the *least* impressive of the available signals and a
careful reviewer will read the wandb log, see `env_reward=0.000` throughout,
and discount the gain.

Plan: after fixing B2, run at least **200 GRPO steps** (Self-Serve Guide
§17 calls this out — "Train long enough that the curves mean something").
On a free T4, with 4-bit Llama-3.2-1B, LoRA rank 16, batch 8, that's about
two hours. If you need to fit it in one session, drop to LoRA rank 8 and
batch 4.

### B5. `client.py` has a bare import that breaks under packaged install

`client.py` line 11: `from models import IncidentCommanderAction, IncidentCommanderObservation`.

When the `incident_commander` package is installed via `pyproject.toml`
(as `pyproject.toml`'s `package-dir = { "incident_commander" = "." }` will
arrange), the `__init__.py` does `from .client import IncidentCommanderEnv`,
which triggers `client.py`, which then runs `from models import ...` —
**not** `from .models import ...`. That works on Colab only because the repo
root is on `sys.path` from the `git clone … && sys.path.insert(...)` dance.
It won't work for anyone who does `pip install incident-commander` from your
HF Space.

Fix: make all intra-package imports relative or fully qualified. Easiest:
change `client.py` to `from .models import ...` and add the same try/except
fallback used in `server/incident_commander_environment.py` (with a real
fallback this time — see B1).

`training.py`, `inference.py`, and `inference_openrouter.py` all have the
same shape (`from models import ...`, `from server.incident_commander_environment import ...`).
They work when run as scripts from the repo root and break under packaged
install. For scripts that's defensible; for `client.py` it isn't.

### B6. `training.py` is unused and untested

`HACKATHON_MEMORY` lists `training.py` as "reference, Colab is actual
execution." The actual training pipeline is the embedded `SyncEnvClient` in
the notebook. `training.py` itself imports `from client import IncidentCommanderEnv`
and tries to drive the env via the async EnvClient — exactly the path your
team documented as broken under GRPO ("openenv EnvClient is async — doesn't
work with GRPO trainer").

If a judge runs `python training.py --run` on a fresh checkout (the README
says they can), it will fail. Either:

- delete `training.py` and update the README to point at the Colab
  notebook as the canonical training entrypoint, **or**
- fix `training.py` to use the same multi-turn rollout shape from B2 and
  verify it runs end-to-end before submission.

### B7. `project_report_complete.md` is referenced but absent

`HACKATHON_MEMORY.md` cites `project_report_complete.md` twice:

> Storytelling: project_report_complete.md ready
> Prepare presentation/demo using project_report_complete.md

It is not in the repo. Storytelling is 30 % of the score. Either commit
that file or accept that the README + the < 2 min video are the entire
storytelling surface.

---

## Important fixes (lower priority but real)

### I1. Expose the four reward signals to GRPO as separate `reward_funcs`

Right now both `training.py` and the notebook pass
`reward_funcs=[environment_reward_func, format_reward_func]`. The four
signals defined in `rewards.py` (`service_recovery`, `root_cause_accuracy`,
`action_quality`, `speed`) are collapsed into one scalar inside
`compute_total_reward`. TRL's `GRPOTrainer` accepts a list — each function
is logged independently. Refactor to four wrappers and you get four
columns in your wandb log + four lines in your README plot, which is
exactly what Criterion 3 wants.

### I2. Don't floor the aggregate at 0

`rewards.py` line 178: `clamped = max(0.0, min(1.0, total / MAX_RAW))`.

Once the agent is doing the wrong thing twice, all rollouts sit at 0.0 and
GRPO advantage collapses. Use signed `total / MAX_RAW` directly and let
advantages span [-1, 1]. Pair with a small **shaping bonus** for first-time
visits to phases (first `read_logs` on the root service: +0.1, first
`identify_cause` regardless of correctness: +0.1, first `monitor_recovery`:
+0.1) so an untrained Llama-3.2-1B can stumble into positive trajectories.
That's the "probability of a good answer is greater than zero" condition
from the Self-Serve Guide §15.

### I3. `inference.py` regex is buggy

```python
match = re.search(r"<action>(.*?):?(.*?)</action>", completion)
```

Lazy `.*?` plus optional `:?` plus another lazy `.*?` causes group 1 to
match the empty string and group 2 to swallow the rest. Real Llama
completions like `<action>read_logs:payment-service</action>` parse to
`("", "")`. The notebook and `training.py` use the better
`r"<action>\s*([^:<]+)(?::([^<]+))?\s*</action>"` — copy that into
`inference.py` (and add `re.DOTALL` so newlines inside the tag don't break
parsing).

### I4. Both inference scripts silently inject defaults

`inference_openrouter.py` and `inference.py` both do
`if not target: target = "payment-service"` and (in the OpenRouter one) a
forced `read_logs:payment-service` step when the model emits nothing
parseable. That gives the model unearned credit during evaluation. At
difficulty 1 the scenario root service is `payment-service` only ~half the
time (the bank also includes `api-gateway`, `inventory-service`,
`auth-service`), so the default is wrong as often as right. Replace both
with a tracked `parse_failures` counter and either skip the step or end
the episode.

### I5. Curriculum doesn't actually pass through

`Simulator(difficulty=…)` is on the server. The notebook's
`SyncEnvClient.reset()` posts `{}` — no difficulty in the body. So whatever
`difficulty` was passed to `IncidentCommanderEnvironment(...)` once on the
server is what every client sees forever. To do real curriculum (Reward FAQ
§14):

1. `IncidentCommanderEnvironment.reset(difficulty=...)` already accepts the
   arg — good.
2. Make the server's `/reset` handler forward a `difficulty` field from the
   request body to that method.
3. Make `SyncEnvClient.reset(difficulty)` send it.
4. In the training loop, bump difficulty when a moving-average reward
   crosses a threshold.

### I6. Pin dependency versions

`requirements.txt` and `pyproject.toml` both pin only minimum versions.
Reward FAQ §59.6 calls out version skew as a known Unsloth/TRL pain point.
Pin the exact versions you trained against, or judges who re-run your Colab
will get something different from what you saw.

Also: `requirements.txt` mixes runtime (`fastapi`, `uvicorn`) with training
(`unsloth`, `torch`) and test (`pytest`) dependencies. The HF Space Docker
image will install everything including unsloth+torch, which fights with
the slim `python:3.11-slim` base and bloats the image to tens of GB.
`server/requirements.txt` already has the right runtime-only set — just
delete `requirements.txt` at the root and let `pyproject.toml`'s
`[project.optional-dependencies] dev` carry the test deps.

### I7. Reward-hacking pre-check

Reward FAQ §57: "Do not optimize a reward you have not tried to break
yourself first." Two specific hacks worth pre-empting:

1. Read the same service three times via `read_logs`/`read_metrics` to pad
   `MIN_INVESTIGATION_STEPS`, then guess the most common root cause. Track
   *unique service-tool pairs*, and require at least one `read_logs`
   *on the root service* before `identify_cause` is honoured.
2. Spam `monitor_recovery` after the first fix; the env keeps returning
   the cached `post_fix_status`. If `post_fix_status == "recovered"` is
   reachable from the wrong fix at any rate, the agent will learn to spam.
   Quick fix: only set `post_fix_status` when the *correct* fix matches
   the *true* root cause AND a hypothesis is locked.

### I8. README hygiene

- Add a Space link near the top: `https://abishek-priyan-369-incident-commander.hf.space`.
- Embed `before_after_comparison.png` and `training_curves.png` (both saved
  by the notebook) under a "Results" section. Judges Guide is explicit:
  "Embed the key plots in your README with a one-line caption."
- Add a small architecture diagram (services → root cause → reward signals).
- Link to the < 2 min video / mini-blog once they exist (minimum
  submission requirement).
- Remove or update the "outputs/commander_final" reference — that path
  isn't created by the current notebook in a clean run.
- Drop `venv/` from the repo. It's ~50 MB of binary noise and `.dockerignore`
  doesn't exclude it from the HF Space build context.

---

## What the latest Colab notebook is doing right (worth keeping)

1. **Real `before_metrics`.** The latest `colab/incident_commander_training.ipynb`
   computes `before_metrics` from a real `evaluate_model(...)` call — the
   hardcoded literal that appeared in `trained_halfway (1).ipynb` is gone.
   That fixes the credibility issue I flagged in the first draft.
2. **Defensive checkpointing.** "Run 10 steps at a time, save checkpoints
   between runs" is exactly the right strategy for free-tier Colab and
   matches the lesson from Run 1's lost weights.
3. **`VALID_ACTIONS` whitelist** in `environment_reward_func` catches the
   `action_name` / `lock-hypothesis` placeholder text the model emits early
   in training, so the env doesn't waste HTTP calls on garbage.
4. **`fast_inference=False`** with the comment "vLLM crashes on T4" — that's
   a real issue and worth keeping the workaround documented.

---

## What's strong about the project itself

- The environment design genuinely fits Theme 3.1: 8 root causes × dynamic
  log/metric values × red-herring services × multi-hop causal chains × an
  enforced *investigate → hypothesise → fix → monitor → resolve* loop. That
  is precisely the "real interaction with tools, APIs, or dynamic systems
  where the model is expected to do real hard work instead of exploiting
  short-cuts" wording from the theme description.
- The 4-signal anti-shortcut reward design is well-thought-through on
  paper — it just needs an actual multi-turn rollout to exercise it.
- Scenario bank breadth (OOM, bad deploy, pool exhaustion, Redis cascade,
  leak, traffic spike, config error, expired cert) is wider than most
  hackathon submissions.
- `tests/smoke_test.py` and `tests/test_env.py` cover the right things:
  100-call reset stability, partial observability, anti-shortcut blocks,
  multi-round fix loop, scenario randomisation, reward functions.
- README's action schema and root-cause→fix table are the kind of crisp
  documentation Judges Guide calls "tell a story, not an API doc."
- `server/app.py` mounts a Gradio demo at `/web` — that's a nice touch
  for storytelling: judges can manually click through an episode.

---

## Honest assessment

The environment is competitive on Criterion 1 (Innovation, 40 %). Once
B2 is fixed and you have one honest 200+ step training run with the four
reward components logged separately, it's also competitive on Criterion 4
(Reward & Pipeline, 10 %). Criterion 3 (Showing Improvement, 20 %) is the
weakest right now — the only actual "improvement" your team has captured
is format compliance going from 10 % to 25 %, and `env_reward=0.000`
through every step you trained will read poorly to a careful judge.
Criterion 2 (Storytelling, 30 %) is salvageable: write the README results
section, embed the plots, link a 90-second video.

Highest-leverage moves, in order:

1. Fix B2 (reward function shape). Without this, the rest of the
   improvements are window-dressing.
2. Run 200+ steps. Capture wandb log.
3. Fix B1 and B5 so the package is shippable.
4. Update the README with embedded plots, the Space URL, and a short
   "What changed after training" section.
5. Record the 90-second video.

The bones are good. The training pipeline is the bottleneck and it is
fixable in well under a day.
