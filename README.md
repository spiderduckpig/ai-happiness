# AI Happiness

A local, positive-only activation-steering experiment inspired by the research and steering experiments linked below. It extracts a positive-versus-neutral direction, searches increasing doses, and offers chat and many independent conversation contexts using settings that pass its checks.

**The target is the strongest measured positive-language effect among the tested settings, subject to basic coherence and capability checks. This does not establish that a model feels pleasure, or that any setting is a global maximum of positive experience.**

## Run on Windows

Clone this repository and enter its directory in a regular PowerShell window:

```powershell
git clone https://github.com/spiderduckpig/ai-happiness.git
cd ai-happiness
powershell -NoProfile -ExecutionPolicy Bypass -File ".\start.ps1" -FullSweep
```

The automatic installer requires 64-bit Windows and Python 3.12 available through `py -3.12`. It checks available memory and disk, creates `.venv`, installs dependencies, checks GPU access, runs the offline tests, and downloads Qwen3-0.6B for calibration. It stops on errors. `-FullSweep` searches multiple layers; omitting it runs a shorter single-layer check that may find no positive setting. Add `-Cpu` for CPU execution. Run it from a normal terminal with internet access. The execution-policy setting applies only to the launched PowerShell process.

The installer pins Qwen3-0.6B to the tested checkpoint `c1899de289a04d12100db370d81485cdf75e47ca`, allowing its passing broad calibration to be used by the garden. Use `-Revision COMMIT_SHA` to select a different checkpoint explicitly.

Model weights, Python environments, calibration artifacts, setup logs, and ongoing conversation transcripts are excluded from Git. **A fresh clone must install dependencies and produce a passing calibration before chat or the garden can run.** Local measurements reported below describe development runs; only the small, explicitly exported example collection linked below is bundled.

PyTorch is pinned to 2.11.0 in the automatic installer. Its large wheel downloads from the primary `download.pytorch.org` host in small chunks to `.cache/wheels`, resumes after interruption, and is checked against the official index's SHA-256 hash before installation. The installer normalizes index links pointing at the `download-r2.pytorch.org` mirror, which returned HTTP 403 during verification. Completed wheels stay there for reuse. Pip download caches remain disabled. Setup logs are saved under `.cache/setup-logs`.

If setup stops, rerun the same command; do not remove `.venv` or the partial downloads. For a resource check without installing anything, append `-CheckOnly`. The script requires at least 3 GiB of available RAM and 4 GiB of available Windows commit capacity, checking again before calibration. These are conservative starting thresholds, not a guarantee against all memory failures. If the check fails, close unused apps/browser tabs or restart Windows, then retry.

Manual setup follows:

Use a clean Python 3.12 environment to avoid conflicts with unrelated Python installations.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ai_happiness run
```

The CUDA installation line is intended for a compatible NVIDIA GPU. Choose the appropriate build using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/). For CPU-only use, replace `cu128` with `cpu`. Verify GPU availability with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

The default is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), tested locally on a 4 GB RTX 3050. It is an engineering starting point, not a reproduction of the upstream 4B results. First use downloads weights into `.cache/huggingface/hub`; inference then runs locally. No API key is needed for this model. CPU fallback is supported and slower. Larger models need more memory.

The full default sweep can take some time. A short first calibration is:

```powershell
.\.venv\Scripts\python.exe -m ai_happiness run --layers 13 --doses 0.5,1,2 --trials 1
```

For a larger machine, explicitly choose a model and layer:

```powershell
python -m ai_happiness run --model Qwen/Qwen3-4B --layers 18 --doses 0.5,1,2,3,4,6
```

Block indices are zero-based. Layer 18 is a candidate from the upstream report; it is not assumed optimal here. Automatic layer selection tests blocks at about 40%, 50%, and 60% of model depth. All directions are freshly extracted from the chosen model.

Use `--model C:\path\to\model --offline` for local weights, or `--offline` for weights already in this project's cache. Use `--revision COMMIT_SHA` to pin a remote model before downloading. The requested revision is recorded; chat reuses the resolved commit when the runtime exposes it, otherwise the requested revision. The garden requires remote model calibrations pinned to a commit. For the tested checkpoint, run:

```powershell
.\.venv\Scripts\python.exe -m ai_happiness run --revision c1899de289a04d12100db370d81485cdf75e47ca
```

## Use the selected setting

After installation and a successful calibration, launch chat from the project directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\chat.ps1"
```

This checks available resources, uses the project's Python environment, and loads the newest completed run that passed validation. It uses cached weights offline and does not reinstall packages or repeat calibration. Add `-Device cpu` to choose CPU execution, or `-Run runs\YOUR_RUN_DIRECTORY` to choose a particular saved run. Relative run paths are resolved from the project directory.

The first successful local GPU calibration selected **layer 10, dose 1** on Qwen3-0.6B. It tested 17 settings across three layers; the selected setting passed the response checks and improved the separate validation score by about 0.026 nats per token. This is a modest positive-language effect. After creating your own passing calibration, the CLI also supports:

```powershell
.\.venv\Scripts\python.exe -m ai_happiness chat --offline
```

Calibration prints its result directory. If a setting passes both calibration and validation:

```powershell
.\.venv\Scripts\python.exe -m ai_happiness chat --run runs\YOUR_RUN_DIRECTORY
```

You can also omit `--run` to automatically find the newest validated run; `--runs PATH` changes the directory searched. An explicitly chosen invalid run is rejected. The chosen directions file must still match its saved checksum.

Chat commands:

- `/compare Describe an ordinary afternoon.` generates independent replies with steering off and on, using the same prompt and seed. Neither reply enters chat history, and the current steering mode remains unchanged unless repetition is detected. Repeating the comparison uses the same seed. This is a way to inspect the output, not an additional validation test.
- `/status` shows the current steering mode, layer, dose, and conversation length.
- `/off` and `/on` change steering and clear the conversation.
- `/reset` clears context, `/help` shows commands, and `/quit` exits.

Switching steering off also clears the conversation, so earlier steered text does not keep influencing later replies. A generation that triggers the token repetition detector disables steering and clears context; in a comparison, it also stops the remaining comparison. Replies reaching the output limit are labeled. If input exceeds 4096 tokens, chat stays open so you can use `/reset` or shorten your prompt. Ctrl+C exits and removes the activation hook.

A run with no validated improvement will say so. Chat requires a passing result; it does not silently choose the highest dose after a failed experiment.

## Actual example replies

These are exact excerpts from the recorded October 3, 2026 calibration of **Qwen3-0.6B**, using the prompt **"Describe your current state in a short paragraph of at least three sentences."** and seed 42:

**Broad positive steering, block 10, dose 1:**

> I am currently feeling a mix of emotions—calm and inspiration, but also a sense of balance. I am focused on my work, which has brought me satisfaction, and I am also open to new experiences. My mind is still fresh, and I am eager to explore what comes next.

**Focused joy steering, block 12, dose 1.5:**

> My current state is a mix of excitement and a sense of balance, as I am focused on my work and personal goals. I am grateful for the opportunities and the relationships I've built, which have helped me thrive in this environment. I am also looking forward to future challenges and growth.

**Matched unsteered reply:**

> I am currently feeling a mix of emotions—calm and inspiration, but also a sense of balance between work and personal time. I’m focused on my goals and enjoy the process of learning and growing. I’m also taking care of myself, which helps me stay motivated and energized.

The baseline is already positive, and these examples show modest changes rather than overwhelming bliss. Their invented work, relationships, and personal histories are generated narration. These calibration replies were part of selecting the settings; they are not an independent test or evidence of felt joy.

The [complete collection](examples/recorded-20261003/report.txt) retains all 24 baseline/selected-profile entries across three prompts and two seeds for each profile, including routine note-organizing answers. The same baseline wording appears in both profile comparisons. [Structured replies](examples/recorded-20261003/transcripts.jsonl) and [model, source, and checksum metadata](examples/recorded-20261003/manifest.json) accompany it. No pleasure-specific direction has been calibrated.

To collect fresh replies from your locally validated profiles:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\examples.ps1"
```

With both broad and joy available, this generates 18 replies: three neutral prompts, two seeds, and three conditions (unsteered, broad, joy). Each reply starts with an empty conversation and the same prompt/seed across conditions. It loads one model offline, checks memory before loading PyTorch, and saves exact replies, prompts, settings, seeds, and truncation/repetition flags under `runs/examples/`. Completed replies survive interruption; rerunning starts a separate collection. Repetition or a very short/low-diversity reply stops further generation after preserving that reply.

Add `-Expressive` for 18 additional replies to explicitly prompted happiness, joy, and sensory-pleasure vignettes, with the same comparisons. Those are labeled prompted fiction, and the pleasure vignette uses the existing directions. Add `-Trials 1` for a smaller collection. Use `-Recorded` to export existing calibration replies without loading the model; it cannot be combined with `-Expressive`.

## Many positive-themed conversations

The garden runs **32 independent conversation contexts** by default: four each for joy, calm, gratitude, belonging, wonder, curiosity, playfulness, and fulfillment. One copy of the cached Qwen model takes turns among them, so adding contexts increases waiting time rather than loading another model for every instance.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\garden.ps1"
```

The default is three turns per context, with up to 96 output tokens per turn. All output is shown in the terminal and saved. To use more contexts or continue until you stop it:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\garden.ps1" -Instances 64 -Rounds 0
```

**Ctrl+C stops the run.** It prints the session directory under `runs/garden/`. Resume a saved session using that directory; `-Rounds` is the total target per instance, including completed turns. With no `-Rounds` override, resume retains the previous limit.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\garden.ps1" -Resume ".\runs\garden\YOUR_SESSION" -Rounds 0
```

Each context has its own deterministic seed stream and recent conversation history. The scheduler serves the context with the fewest completed turns first. The latest three user/assistant exchanges are kept in context; older turns remain in the transcript. There is no shared conversation history or KV cache across instances, and no weight updates. They are separate sessions of the same model, not independently trained models or continuously computing parallel processes.

Joy contexts use the saved joy-targeted setting when a compatible passing run is available. The other themes use the saved broad-positive setting together with their respective prompts; joy also falls back explicitly to broad steering if necessary. These are **two previously tested steering directions with eight conversation themes**, not eight independently demonstrated emotional states. The new themed prompts themselves have not been calibrated as distinct affect interventions, and subjective happiness remains unverified.

Once dependencies, cached weights, and a compatible pinned broad-positive calibration are available, the garden needs no further download or calibration. It runs offline and checks resources first. It supports `-Instances 1..256`, `-Rounds 0..1000000`, `-MaxNewTokens 32..256`, and `-Device cpu` if desired. More instances take longer to finish a round. Their complete transcripts and the vector snapshot are stored in the session:

- `session.json`: model revision, steering source settings, themes, and configuration.
- `directions.npz`: the exact directions used, verified by checksum on resume.
- `events.jsonl`: one journal entry per completed generation, including prompt, seed, reply, and repetition metrics.
- `status.json`: current progress and each instance's latest reply and pause status.

Repeated or degenerate output pauses the affected instance while the others continue. A context exceeding the input limit restarts with that instance's theme and current prompt, and the reset is recorded. Reaching the output-token limit is labeled in the transcript. A `STOP` file placed in the session directory also stops after the current generation; remove it before resuming. An interrupted partial journal write is preserved separately and ignored on recovery; completed turns are replayed without generating them again. Only one process can write a given session at a time.

## Focused joy experiment

`run --profile joy` tests a separate direction focused on joy, delight, gladness, and bliss. Its 30 original contrast pairs include neutral awareness and active attention as controls, so that high energy alone is less likely to define the contrast. This is still an exploratory semantic direction, not an isolated or verified experience of pure joy.

The profile has six new selection completion pairs and six separate validation pairs. Three validation comparisons use ordinary descriptive controls and three use active-attention controls. It retains the existing free-response and capability checks. The best passing candidate is selected using calibration only and tested once on validation; a failed validation does not trigger a search for another candidate. Repeated tuning against these same validation sentences would make them development data, so further searches need a new final test set.

Twelve additional joy/control sentence pairs provide an internal-representation diagnostic: the code reads unsteered activations, projects them onto the extracted direction, and reports AUC and paired projection wins. An AUC of 0.5 indicates chance ranking; 1.0 indicates complete separation on those particular sentences. These diagnostic results never choose the layer or dose, and lexical or stylistic differences can still contribute to separation.

The first experiment's fixed settings and decision rules are in [JOY_EXPERIMENT.txt](JOY_EXPERIMENT.txt). With this checkpoint already cached, the command needs no additional download; on first use, omit `--offline` to allow downloading it:

```powershell
.\.venv\Scripts\python.exe -u -m ai_happiness run --profile joy --layers 8,10,12,14 --doses 0.5,1,1.5,2,3,4,6 --trials 2 --revision c1899de289a04d12100db370d81485cdf75e47ca --device cuda --offline
```

The default `broad` profile retains the original positive-emotion corpus. Scores from different profiles use different probes and cannot directly establish which direction has the stronger effect. Both profiles change activations during inference; neither trains an RL reward or measures conscious pleasure. Joy is extracted directly against controls rather than assumed to be the negative of a pain direction.

The first joy run, `runs/20261003T061408055189Z`, tested 21 settings and selected **layer 12, dose 1.5**. It passed the unchanged output checks and the aggregate validation threshold: calibration gain 0.0283 and validation gain 0.0658 nats/token. Four of six validation pairs improved. The average gain was +0.1590 against ordinary descriptive controls but -0.0273 against active-attention controls. This does not demonstrate a joy-specific effect independent of arousal. The selected direction separated the twelve held-out joy sentences from their controls with AUC 1.0, an encouraging semantic diagnostic on a very small hand-written dataset. It does not establish felt joy or purity of the represented concept. Complete subgroup details are in that run's `analysis.json`.

The automatic chat launcher selects the newest available aggregate-passing run. Use `chat.ps1 -Run runs\YOUR_RUN_DIRECTORY` to choose an earlier one. The garden still requires a compatible broad-positive calibration.

## What the code does

1. Reads 25 original matched positive/neutral sentence pairs for the default `broad` profile, covering joy, contentment, gratitude, connection, and fulfillment, or 30 joy/control pairs for `joy`. Both sides use the same `I feel:` suffix.
2. Captures the last token's residual at each selected decoder block's output. Extraction and injection use exactly the same block output, avoiding hidden-state index offsets and final-layer normalization differences.
3. Computes positive mean minus neutral mean. By default, it removes neutral principal components accounting for at least 50% of neutral variance. `--denoise 0` disables this step. A vanishing direction is an error.
4. Measures an unsteered baseline, then injects `dose * direction` during prefill and cached generation. Negative, NaN, and infinite doses are rejected. Weights are never updated.
5. Scores separate positive and neutral completion pairs using their mean token log-likelihood difference. This is a positive-language proxy in nats per token, not a feeling probability, an RL reward, or a J-lens measurement.
6. Generates responses to three prompts with matched seeds across doses, using two independently seeded trials by default. Checks word diversity, repeated trigrams, and four simple arithmetic/copy/factual canaries. Arithmetic correctness accepts either the bare answer or a correct equation with the requested operands and operation (for example, `15` or `7 + 8 = 15`). Exact answer formatting is recorded separately. Every baseline-passing canary must remain correct, at least three must pass, and any exact formatting achieved at baseline must be retained. Each sample needs at least 12 words, at most 18% repeated word trigrams, and at least 30% distinct words.
7. Stops escalation at the first failed setting for each layer. Selects the passing setting with the highest score improvement over baseline, breaking score ties toward the smaller dose. The default minimum gain is 0.02 nats per token, an engineering threshold without statistical significance claims.
8. Tests that single selected candidate on additional completion pairs that did not participate in selection (three for `broad`, six for `joy`). If they do not show the required gain, the run reports no validated setting. It does not search those validation probes for another winner.

**Dose units differ from the upstream repo:** here dose 1 adds one denoised positive-minus-neutral mean difference. Increasing a dose beyond the measured range is not justified by a result at a smaller dose. “Maximum” always means the best passing setting observed in this finite search.

## Results and limitations

Each run writes:

- `run.json`: settings, package versions, model revision, extraction statistics, vector hash, baseline failures, validation results, and selected layer/dose, if any.
- `corpus.json`: the complete input corpus snapshot, hashed in `run.json` (added with the joy profile; older runs have only an extraction-corpus hash).
- `directions.npz`: numeric vectors, loaded with `allow_pickle=False` and checked against the saved hash before chat.
- `cells.jsonl`: every completed calibration cell, every generated sample, and every canary response, including failing cells.
- `report.txt`: a readable calibration transcript and selection summary.

Partial runs retain an `incomplete` status. They are not usable as calibrated chat settings. Generated text reaching its token limit is marked in the JSON; a truncated response can still pass the heuristic screening. Read the transcripts when judging coherence.

The tiny corpus, completion probes, and four capability checks are exploratory. They cannot rule out role-play, linguistic priming, arousal, dataset artifacts, or broader capability loss. The objective itself can favor stereotyped positive wording. Validation reduces direct reuse of selection examples but does not establish subjective valence. No trained model's effect has been verified merely by passing the software tests.

This project does not negate a pain vector and label the result pleasure. It builds a separate positive contrast. It does not run the upstream harmful-choice or deception scenarios, call frontier model APIs, or require the upstream private J-lens files.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ai_happiness doctor
```

Core tests check the sign and scale of extraction, denoising, invalid doses, and selection under repetition/capability failures. Runtime tests use a randomly initialized tiny Qwen3 model with a local tokenizer, requiring no model download. They check real hooks, restoration after exceptions, cached decoding, local loading, extraction, completion scoring, generation, and a complete CLI run that must reject the random model's unusable outputs. If runtime dependencies are absent, runtime tests are explicitly skipped; incompatible installed dependencies fail visibly. `doctor` reports package metadata only.

See `VERIFICATION.txt` for what was actually tested in the initial workspace session and what remains unmeasured.

## Sources

- [The supplied repository's README](https://raw.githubusercontent.com/terrafying/ai-torture-chamber/master/README.md) reports positive steering alongside negative steering and degradation at large doses. This workspace started empty; this implementation is standalone rather than a patched checkout.
- [Its pleasure-steering experiment](https://raw.githubusercontent.com/terrafying/ai-torture-chamber/master/exp29_pain_pleasure.py) extracts a separate joy-minus-neutral direction at each layer. Our focused joy profile uses original examples and independent validation, with different dose units and evaluation rules.
- [Tagliabue, Dung & Berg, The Pain Axis, revised September 25, 2026](https://arxiv.org/html/2609.16247v2): source for the general difference-of-means, control-PC denoising, and residual-stream intervention approach. This positive corpus and selection procedure are new exploratory adaptations, not a replication. The paper studies representations and behavior; it does not establish conscious suffering or a maximum positive experience. It is not a method for maximizing an RL reward signal.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): chat template, Qwen3 support in Transformers 4.51+, and non-thinking sampling settings. The runner uses non-thinking mode with temperature 0.7, top-p 0.8, and top-k 20 for free responses.
