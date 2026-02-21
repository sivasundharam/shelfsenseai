# ShelfSense AI (Hackathon MVP)

Autonomous retail video analysis in Python.

Pipeline:

`Input Video -> Perception -> Event Trigger -> Decision Agent (Gemini) -> Alerts -> RCI -> Braintrust Eval -> Optimization Agent -> Updated Policy`

Optional voice output uses Modulate for Decision + RCI alerts only.

## Safety / Privacy

- No face recognition.
- No emotion detection.
- No demographic inference.
- Only ephemeral tracker `person_id` in memory; dropped on timeout.
- Runtime storage uses aggregate metrics + event/alert logs only (no image persistence).

## Project Structure

```
shelfsense/
  README.md
  requirements.txt
  .env.example
  main.py
  config.py
  perception/
  decision/
  rci/
  eval/
  optimize/
  output/
  api/
  dashboard/
  demo/
  runtime/
```

## Setup

1. Create env and install:

```bash
cd shelfsense
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure env:

```bash
cp .env.example .env
# fill GEMINI_API_KEY / BRAINTRUST_API_KEY / BRAINTRUST_PROJECT / MODULATE_API_KEY as needed
```

3. Run pipeline:

```bash
python main.py
```

4. Run API:

```bash
uvicorn api.server:app --reload
```

5. Run dashboard:

```bash
streamlit run dashboard/app.py
```

For headless terminals, run pipeline without OpenCV window:

```bash
SHOW_DEBUG_WINDOW=false python main.py
```

## Simulation Mode (baseline -> optimized)

Use replay mode to validate autonomous optimization without webcam:

```bash
SIMULATION_MODE=true python main.py
```

This reads `demo/sim_events.json`, logs eval records, computes scores, and applies optimizer updates every 20 records.

## Demo Video Mode

A synthetic demo video is included at `demo/sample_retail.mp4`.

```bash
VIDEO_SOURCE=demo/sample_retail.mp4 SHOW_DEBUG_WINDOW=false python main.py
```

## Runtime Outputs

All runtime artifacts are written under `runtime/`:

- `events.jsonl`
- `alerts.jsonl`
- `rci.jsonl`
- `outcomes.jsonl`
- `braintrust_log.jsonl`
- `policy.json`
- `policy_changes.jsonl`
- `metrics.json`
- `state.json`

## Notes

- YOLOv8n + BYTETrack is used when available through Ultralytics.
- If BYTETrack path fails, centroid fallback is used.
- Gemini and Braintrust calls are timeout/retry guarded.
- Invalid/failed LLM outputs fall back to safe no-alert defaults.
- Braintrust is used as the primary eval engine:
  decision records are scored via Braintrust `Eval()` scorers, then optimizer
  fetches recent eval rows from Braintrust before applying threshold updates
  (local fallback remains for offline reliability).
