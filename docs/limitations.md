# Limitations

- **Personal-speaker regression.** Every fine-tune regressed on Awi's s02 by
  ~3–6 pt; the release gate (condition 2) therefore said "keep v1" for all three,
  and E1f_s1 was still deployed (the logged deviation) for its group-speaker and
  FAR gains. This trade-off is deliberate and documented in `docs/evaluation.md`.
- **Small, skewed test coverage.** The holdout has only 84 real clips from one
  unseen speaker; real-speaker accuracy (77.8 % for E1f) is materially below
  synthetic (99.7 %). Treat real-speech numbers as the honest ceiling.
- **Single unseen speaker.** Speaker-disjointness is proven for one group test
  speaker (`201435283`) and one holdout speaker (`202520785`); generalisation to
  unseen accents/mics is untested.
- **Non-commercial data.** FSC's non-commercial terms propagate to the weights
  (CC BY-NC 4.0); the models cannot be used commercially without re-licensing
  the data.
- **Single-microphone, near-field.** The model is validated on a laptop mic and
  the Pi's default mic; far-field or noisy deployment is out of scope.
- **VAD disabled.** Endpointing is implemented but off by default; long-utterance
  behaviour and hangover under real noise remain lightly tested.
- **No cloud/ASR in the loop.** The command path is audio → intent + slot only;
  no transcription or LLM, by design (this bounds capability but guarantees
  on-device latency and privacy).
