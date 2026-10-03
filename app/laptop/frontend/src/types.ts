export type Prediction = {
  model_id: string;
  class_key: string;
  intent: string;
  slot: string | null;
  confidence: number;
  top_k: [string, number][];
  feature_ms: number;
  inference_ms: number;
};

export type LastResult = {
  command_id: string;
  source: "live" | "replay" | "simulated";
  primary: Prediction;
  accepted: boolean;
  reject_reason: string | null;
  threshold: number;
  compare: Prediction[];
  members: Prediction[];
  clip_ms: number;
  clip_rms_dbfs: number | null;
  t_capture_end_ms: number;
  t_result_ms: number;
  received_ms: number;
  action_ms: number;
  expected_class: string | null;
  outcome: { executed: boolean; action: string; target: string | null; detail: string | null };
};

export type ScoreEntry = {
  dataset_id: string | null;
  split: string | null;
  metrics: Record<string, number>;
  threshold: number | null;
  source: string | null;
  note: string | null;
  status: "reported" | "not_evaluated" | "not_recorded" | "not_applicable";
};

export type ModelSummary = {
  id: string;
  display_name: string;
  description?: string | null;
  based_on?: string | null;
  engine: string;
  members?: string[];
  n_classes: number;
  quantization: string;
  architecture: Record<string, unknown>;
  lineage: Record<string, string | null>;
  offline_metrics: Record<string, number>;
  scores?: { training: ScoreEntry; validation: ScoreEntry; test: ScoreEntry };
  min_confidence: number;
  window_s: number;
  status: "ready" | "not_loaded" | "unavailable" | "invalid";
  detail: string | null;
  size_bytes: number | null;
  sha256_12: string | null;
  params: number | null;
};

export type WakeInfo = {
  id: string;
  display_name: string;
  kind: string;
  status: string;
  detail: string | null;
  phrase: string | null;
  threshold: number | null;
  patience: number | null;
  debounce_s: number | null;
};

export type Media = {
  status: "playing" | "paused" | "stopped";
  level: number;
  max_level: number;
  level_pct: number;
  ducked: boolean;
  track_index: number;
  track_count: number;
  track_title: string;
  backend: string;
  backend_ok: boolean;
  backend_detail: string | null;
};

export type EdgeState = {
  phase: string;
  window_s: number;
  active_vcm: string;
  compare_ids: string[];
  active_wake: string;
  threshold_override: number | null;
  models: ModelSummary[];
  wake_engines: WakeInfo[];
  pre_capture_delay_s: number;
  edge_chime: boolean;
  endpointing: string;
  media: Media;
  health: {
    device: string;
    uptime_s: number;
    audio_stream: boolean;
    level_dbfs: number | null;
    cpu_percent: number | null;
    process_rss_mb: number | null;
    system_mem_used_mb: number | null;
    vcm_ready: boolean;
    wake_ready: boolean;
    media_ok: boolean;
    vad: string;
    last_error: string | null;
  };
};

export type TimelineEvent = {
  type: string;
  origin: "edge" | "laptop";
  ts_ms: number;
  command_id: string | null;
  payload: Record<string, any>;
  ts_edge_ms?: number;
  t_received_ms?: number;
  timing_valid?: boolean;
  clock_quality?: ClockQuality;
};

export type ClockQuality = "local" | "accurate" | "approximate" | "syncing" | "unknown";

export type CommandRow = {
  command_id: string;
  ts_ms: number;
  source: string;
  model_id: string;
  class_key: string;
  confidence: number;
  accepted: number;
  reject_reason: string | null;
  inference_ms: number;
  capture_to_result_ms: number;
  result_to_action_ms: number;
  action: string;
  executed: number;
  expected_class: string | null;
};

export type ScoreRow = {
  model_id: string;
  n_clips: number;
  n_labelled: number;
  n_correct: number;
  n_intent_correct: number;
  accuracy: number | null;
  intent_accuracy: number | null;
  mean_conf: number | null;
  mean_inference_ms: number | null;
};

export type AppState = {
  server_ms: number;
  link: { mode: string; connected: boolean; url: string; rtt_ms: number | null };
  edge_clock: {
    offset_ms: number; ready: boolean; n_samples: number; rtt_ms: number | null; spread_ms: number | null;
    has_estimate?: boolean; source?: "local" | "ntp" | "ingress" | null; quality?: ClockQuality;
    jumps?: number; last_jump_ms?: number | null;
  };
  edge: EdgeState | null;
  phase: { phase: string; command_id?: string; window_s?: number; accepted?: boolean; mode?: "vad" | "fixed" };
  level_dbfs: number | null;
  last_result: LastResult | null;
  mic: { enabled: boolean; status: string; detail: string | null; device_rate: number | null };
  internet_ok: boolean | null;
  devices: {
    light: { power: boolean; brightness: string; brightness_pct: number; color: string | null };
    thermostat: { setpoint: string; setpoint_c: number; options_c: number[]; mode: string };
    timer: { status: string; label: string | null; duration_s: number; ends_at_ms: number | null };
    alarm: { status: string; slot: string | null; at_iso: string | null };
    reminders: { items: { id: number; text: string; source: string; done: number }[]; focus_until_ms: number };
    phone: {
      contact: { name: string; phone: string };
      call_status: string;
      call_started_ms: number | null;
      messages: { text: string; ts_ms: number; to: string }[];
    };
    weather: Weather | null;
    weather_last: Weather | null;
    toast: { kind: string; at_ms: number; tz?: string } | null;
  };
  commands: CommandRow[];
  scoreboard: ScoreRow[];
  benchmark: BenchmarkSnapshot;
  display_names: {
    models: Record<string, { name: string; based_on: string; description: string; role?: string }>;
    datasets: Record<string, { name: string; description: string; display_id?: string }>;
    sessions: Record<string, { name: string; role: string }>;
  };
  ontology: { leaf_labels: string[]; slot_values: Record<string, string[]>; schema_version: string };
  timeline: TimelineEvent[];
};

export type Weather = {
  available: boolean;
  cached?: boolean;
  cached_note?: string;
  provider?: string;
  provider_url?: string;
  endpoint?: string;
  mode?: "live" | "cached_fixture" | "unavailable";
  observed?: string;
  temperature_c?: number;
  apparent_c?: number;
  humidity_pct?: number;
  wind_kmh?: number;
  condition?: string;
  location?: string;
  error?: string;
};

// ---- live benchmark ------------------------------------------------------
export type BenchmarkSnapshot = {
  running: boolean;
  session_id: string | null;
  phase: string;
  locked: { model_id: string | null; threshold: number | null } | null;
  trial_idx?: number;
  trial_total?: number;
  prompt?: { kind: string; say: string; expected_class: string | null } | null;
  last_outcome?: { outcome: string; predicted: string; confidence: number; inference_ms: number } | null;
  counts?: Record<string, number>;
  false_wakes?: number;
};

export type BenchRate = { n: number; k: number; value: number | null; lo: number; hi: number };
export type BenchMs = { n: number; p50: number | null; p95: number | null };
export type BenchChip = {
  key: string; label: string; value: number | null; lo: number | null; hi: number | null;
  n: number | null; target: number; status: "pass" | "fail" | "too_few";
};
export type BenchSessionRow = {
  session_id: string; alias: string; set_type: string; model_id: string;
  cond_noise: string; cond_distance: string; created_ms: number; scored: number;
  correct_rate: number | null; wrong_rate: number | null; neg_ignored: number;
  neg_total: number; wake_miss: number; model_median_ms: number | null; e2e_median_ms: number | null;
};
export type BenchSummary = {
  pooled: {
    correct: BenchRate; not_understood: BenchRate; wrong: BenchRate;
    false_accept: BenchRate; wake_miss: BenchRate; no_result: number; false_wakes: number;
    pi_model_ms: BenchMs; pi_features_ms: BenchMs; e2e_ms: BenchMs;
    model: { params: number | null; size_kb: number | null; runtime: string | null; name: string | null };
  };
  sessions: BenchSessionRow[];
  heatmap: { columns: string[]; rows: { session_id: string; alias: string; set_type: string; cells: { col: string; color: string }[] }[] };
  chips: BenchChip[];
};
export type BenchSessionInfo = {
  session_id: string; alias: string; created_ms: number; finished_ms: number | null;
  status: string; mode: string; set_type: string; set_seed: number | null;
  model_id: string; threshold: number; cond_noise: string; cond_distance: string;
  network_offline: number; consent_results: number; consent_audio: number; note: string | null;
};
