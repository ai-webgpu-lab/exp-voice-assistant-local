const EXECUTION_MODES = {
  webgpu: {
    id: "webgpu",
    label: "WebGPU ready",
    backend: "webgpu",
    fallbackTriggered: false,
    workerMode: "worker",
    stageMultiplier: 1,
    intentDelayMs: 22,
    replyDelayMs: 26,
    ttsStreamGapMs: 16
  },
  fallback: {
    id: "fallback",
    label: "CPU fallback",
    backend: "cpu",
    fallbackTriggered: true,
    workerMode: "main",
    stageMultiplier: 1.8,
    intentDelayMs: 42,
    replyDelayMs: 44,
    ttsStreamGapMs: 24
  }
};

function resolveExecutionMode() {
  const requested = new URLSearchParams(window.location.search).get("mode");
  const hasWebGpu = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  if (requested === "fallback" || !hasWebGpu) return EXECUTION_MODES.fallback;
  return EXECUTION_MODES.webgpu;
}

const executionMode = resolveExecutionMode();

const requestedMode = typeof window !== "undefined"
  ? new URLSearchParams(window.location.search).get("mode")
  : null;
const isRealRuntimeMode = typeof requestedMode === "string" && requestedMode.startsWith("real-");
const REAL_ADAPTER_WAIT_MS = 5000;
const REAL_ADAPTER_LOAD_MS = 20000;

function withTimeout(promise, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs);
    promise.then((value) => {
      clearTimeout(timer);
      resolve(value);
    }, (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function findRegisteredRealRuntime() {
  const registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null;
  if (!registry || typeof registry.list !== "function") return null;
  return registry.list().find((adapter) => adapter && adapter.isReal === true) || null;
}

async function awaitRealRuntime(timeoutMs = REAL_ADAPTER_WAIT_MS) {
  const startedAt = performance.now();
  while (performance.now() - startedAt < timeoutMs) {
    const adapter = findRegisteredRealRuntime();
    if (adapter) return adapter;
    if (typeof window !== "undefined" && window.__aiWebGpuLabRealVoiceAssistantBootstrapError) {
      return null;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return null;
}

const state = {
  startedAt: performance.now(),
  environment: buildEnvironment(),
  fixture: null,
  active: false,
  run: null,
  transcript: "",
  reply: "",
  realAdapterError: null,
  logs: []
};

const elements = {
  statusRow: document.getElementById("status-row"),
  summary: document.getElementById("summary"),
  runAssistant: document.getElementById("run-assistant"),
  downloadJson: document.getElementById("download-json"),
  transcriptView: document.getElementById("transcript-view"),
  replyView: document.getElementById("reply-view"),
  metricGrid: document.getElementById("metric-grid"),
  metaGrid: document.getElementById("meta-grid"),
  logList: document.getElementById("log-list"),
  resultJson: document.getElementById("result-json")
};

function round(value, digits = 2) {
  if (!Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function parseBrowser() {
  const ua = navigator.userAgent;
  for (const [needle, name] of [["Edg/", "Edge"], ["Chrome/", "Chrome"], ["Firefox/", "Firefox"], ["Version/", "Safari"]]) {
    const marker = ua.indexOf(needle);
    if (marker >= 0) return { name, version: ua.slice(marker + needle.length).split(/[\s)/;]/)[0] || "unknown" };
  }
  return { name: "Unknown", version: "unknown" };
}

function parseOs() {
  const ua = navigator.userAgent;
  if (/Windows NT/i.test(ua)) return { name: "Windows", version: (ua.match(/Windows NT ([0-9.]+)/i) || [])[1] || "unknown" };
  if (/Mac OS X/i.test(ua)) return { name: "macOS", version: ((ua.match(/Mac OS X ([0-9_]+)/i) || [])[1] || "unknown").replace(/_/g, ".") };
  if (/Android/i.test(ua)) return { name: "Android", version: (ua.match(/Android ([0-9.]+)/i) || [])[1] || "unknown" };
  if (/(iPhone|iPad|CPU OS)/i.test(ua)) return { name: "iOS", version: ((ua.match(/OS ([0-9_]+)/i) || [])[1] || "unknown").replace(/_/g, ".") };
  if (/Linux/i.test(ua)) return { name: "Linux", version: "unknown" };
  return { name: "Unknown", version: "unknown" };
}

function inferDeviceClass() {
  const threads = navigator.hardwareConcurrency || 0;
  const memory = navigator.deviceMemory || 0;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  if (mobile) return memory >= 6 && threads >= 8 ? "mobile-high" : "mobile-mid";
  if (memory >= 16 && threads >= 12) return "desktop-high";
  if (memory >= 8 && threads >= 8) return "desktop-mid";
  if (threads >= 4) return "laptop";
  return "unknown";
}

function buildEnvironment() {
  return {
    browser: parseBrowser(),
    os: parseOs(),
    device: {
      name: navigator.platform || "unknown",
      class: inferDeviceClass(),
      cpu: navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency} threads` : "unknown",
      memory_gb: navigator.deviceMemory || undefined,
      power_mode: "unknown"
    },
    gpu: {
      adapter: executionMode.fallbackTriggered ? "cpu-fallback-audio" : "navigator.gpu available",
      required_features: executionMode.fallbackTriggered ? [] : ["shader-f16"],
      limits: executionMode.fallbackTriggered ? {} : { maxBindGroups: 4 }
    },
    backend: executionMode.backend,
    fallback_triggered: executionMode.fallbackTriggered,
    worker_mode: executionMode.workerMode,
    cache_state: "warm"
  };
}

function log(message) {
  state.logs.unshift(`[${new Date().toLocaleTimeString()}] ${message}`);
  state.logs = state.logs.slice(0, 12);
  renderLogs();
}

async function loadFixture() {
  if (state.fixture) return state.fixture;
  const response = await fetch("./voice-fixture.json", { cache: "no-store" });
  state.fixture = await response.json();
  return state.fixture;
}

function words(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s-]/g, " ").split(/\s+/).filter(Boolean);
}

function chars(text) {
  return text.toLowerCase().replace(/\s+/g, "").split("");
}

function levenshtein(left, right) {
  const rows = left.length + 1;
  const cols = right.length + 1;
  const matrix = Array.from({ length: rows }, () => new Array(cols).fill(0));

  for (let row = 0; row < rows; row += 1) matrix[row][0] = row;
  for (let col = 0; col < cols; col += 1) matrix[0][col] = col;

  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const cost = left[row - 1] === right[col - 1] ? 0 : 1;
      matrix[row][col] = Math.min(
        matrix[row - 1][col] + 1,
        matrix[row][col - 1] + 1,
        matrix[row - 1][col - 1] + cost
      );
    }
  }

  return matrix[left.length][right.length];
}

function renderTranscript(text) {
  elements.transcriptView.textContent = text || "No voice turn yet.";
}

function renderReply(text) {
  elements.replyView.textContent = text || "No reply emitted yet.";
}

function mutateTranscript(reference) {
  if (!executionMode.fallbackTriggered) return reference;
  return reference
    .replace("fallback notes", "fallback modes")
    .replace("lab status", "status");
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function runRealRuntimeVoiceAssistant(adapter) {
  log(`Connecting real runtime adapter '${adapter.id}'.`);
  await withTimeout(
    Promise.resolve(adapter.loadModel({ modelId: "voice-assistant-local-default" })),
    REAL_ADAPTER_LOAD_MS,
    `loadModel(${adapter.id})`
  );
  const prefill = await withTimeout(
    Promise.resolve(adapter.prefill({ promptTokens: 96 })),
    REAL_ADAPTER_LOAD_MS,
    `prefill(${adapter.id})`
  );
  const decode = await withTimeout(
    Promise.resolve(adapter.decode({ tokenBudget: 32 })),
    REAL_ADAPTER_LOAD_MS,
    `decode(${adapter.id})`
  );
  log(`Real runtime adapter '${adapter.id}' ready: prefill_tok_per_sec=${prefill?.tokPerSec ?? "?"}, decode_tok_per_sec=${decode?.tokPerSec ?? "?"}.`);
  return { adapter, prefill, decode };
}

async function runAssistantTurn() {
  if (state.active) return;
  state.active = true;
  state.run = null;
  state.transcript = "";
  state.reply = "";
  renderTranscript("");
  renderReply("");
  render();

  if (isRealRuntimeMode) {
    log(`Mode=${requestedMode} requested; awaiting real runtime adapter registration.`);
    const adapter = await awaitRealRuntime();
    if (adapter) {
      try {
        const { prefill, decode } = await runRealRuntimeVoiceAssistant(adapter);
        state.realAdapterPrefill = prefill;
        state.realAdapterDecode = decode;
        state.realAdapter = adapter;
      } catch (error) {
        state.realAdapterError = error?.message || String(error);
        log(`Real runtime '${adapter.id}' failed: ${state.realAdapterError}; falling back to deterministic.`);
      }
    } else {
      const reason = (typeof window !== "undefined" && window.__aiWebGpuLabRealVoiceAssistantBootstrapError) || "timed out waiting for adapter registration";
      state.realAdapterError = reason;
      log(`No real runtime adapter registered (${reason}); falling back to deterministic voice assistant baseline.`);
    }
  }

  const fixture = await loadFixture();
  const startedAt = performance.now();
  let firstPartialMs = 0;

  log(`Wake word detected: ${fixture.wakeWord}.`);
  for (let index = 0; index < fixture.segments.length; index += 1) {
    const segment = fixture.segments[index];
    await sleep(segment.processingMs * executionMode.stageMultiplier);
    const partial = fixture.segments.slice(0, index + 1).map((item) => item.text).join(" ");
    state.transcript = partial;
    if (!firstPartialMs) firstPartialMs = performance.now() - startedAt;
    renderTranscript(state.transcript);
    log(`Partial ${index + 1}/${fixture.segments.length}: ${segment.text}`);
  }

  const finalTranscript = mutateTranscript(state.transcript);
  state.transcript = finalTranscript;
  renderTranscript(finalTranscript);
  const finalLatencyMs = performance.now() - startedAt;

  const referenceWords = words(fixture.reference);
  const predictedWords = words(finalTranscript);
  const referenceChars = chars(fixture.reference);
  const predictedChars = chars(finalTranscript);
  const wer = levenshtein(referenceWords, predictedWords) / Math.max(referenceWords.length, 1);
  const cer = levenshtein(referenceChars, predictedChars) / Math.max(referenceChars.length, 1);

  const intentStartedAt = performance.now();
  await sleep(fixture.intent.processingMs + executionMode.intentDelayMs);
  const intentMs = performance.now() - intentStartedAt;
  log(`Intent routed: ${fixture.intent.id}.`);

  const replyStartedAt = performance.now();
  await sleep(fixture.reply.processingMs + executionMode.replyDelayMs);
  state.reply = fixture.reply.text;
  renderReply(state.reply);
  const replyMs = performance.now() - replyStartedAt;
  log("Deterministic reply draft prepared.");

  const ttsStartedAt = performance.now();
  let firstAudioMs = 0;
  for (let index = 0; index < fixture.ttsChunks.length; index += 1) {
    const chunk = fixture.ttsChunks[index];
    await sleep((chunk.synthesisMs + executionMode.ttsStreamGapMs) * executionMode.stageMultiplier);
    if (!firstAudioMs) firstAudioMs = performance.now() - startedAt;
    log(`TTS chunk ${index + 1}/${fixture.ttsChunks.length}: ${chunk.text}`);
  }
  const ttsMs = performance.now() - ttsStartedAt;
  const roundtripMs = performance.now() - startedAt;

  state.run = {
    audioSeconds: fixture.audioSeconds,
    segmentCount: fixture.segments.length,
    wakeWord: fixture.wakeWord,
    firstPartialMs,
    finalLatencyMs,
    roundtripMs,
    firstAudioMs,
    audioSecPerSec: fixture.audioSeconds / Math.max(finalLatencyMs / 1000, 0.001),
    wer,
    cer,
    transcript: finalTranscript,
    reply: fixture.reply.text,
    intentId: fixture.intent.id,
    intentMs,
    replyMs,
    ttsMs,
    ttsVoice: fixture.ttsVoice,
    realAdapter: state.realAdapter || null
  };
  state.active = false;
  log(`Voice turn complete: final ${round(state.run.finalLatencyMs, 2)} ms, roundtrip ${round(state.run.roundtripMs, 2)} ms.`);
  render();
}

function describeRuntimeAdapter() {
  const registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null;
  const requested = typeof window !== "undefined"
    ? new URLSearchParams(window.location.search).get("mode")
    : null;
  if (registry) {
    return registry.describe(requested);
  }
  return {
    id: "deterministic-voice-assistant",
    label: "Deterministic Voice Assistant",
    status: "deterministic",
    isReal: false,
    version: "1.0.0",
    capabilities: ["prefill", "decode", "fixed-output-budget"],
    runtimeType: "synthetic",
    message: "Runtime adapter registry unavailable; using inline deterministic mock."
  };
}

function buildResult() {
  const run = state.run;
  return {
    meta: {
      repo: "exp-voice-assistant-local",
      commit: "bootstrap-generated",
      timestamp: new Date().toISOString(),
      owner: "ai-webgpu-lab",
      track: "audio",
      scenario: (state.run && state.run.realAdapter) ? `voice-assistant-local-real-${state.run.realAdapter.id}` : (run ? "voice-assistant-local-readiness" : "voice-assistant-local-pending"),
      notes: run
        ? `wakeWord=${run.wakeWord}; segments=${run.segmentCount}; intent=${run.intentId}; ttsVoice=${run.ttsVoice}; firstAudioMs=${round(run.firstAudioMs, 2)}; roundtripMs=${round(run.roundtripMs, 2)}; backend=${state.environment.backend}; fallback=${state.environment.fallback_triggered}${state.run && state.run.realAdapter ? `; realAdapter=${state.run.realAdapter.id}` : (isRealRuntimeMode && state.realAdapterError ? `; realAdapter=fallback(${state.realAdapterError})` : "")}`
        : "Run the deterministic voice assistant readiness harness."
    },
    environment: state.environment,
    workload: {
      kind: "voice-assistant",
      name: "voice-assistant-local-readiness",
      input_profile: state.fixture ? `${state.fixture.audioSeconds}s-${state.fixture.segments.length}-segments` : "fixture-pending",
      model_id: "deterministic-voice-assistant-local-v1",
      dataset: "voice-fixture-v1"
    },
    metrics: {
      common: {
        time_to_interactive_ms: round(performance.now() - state.startedAt, 2) || 0,
        init_ms: run ? round(run.roundtripMs, 2) || 0 : 0,
        success_rate: run ? 1 : 0.5,
        peak_memory_note: navigator.deviceMemory ? `${navigator.deviceMemory} GB reported by browser` : "deviceMemory unavailable",
        error_type: ""
      },
      stt: {
        audio_sec_per_sec: run ? round(run.audioSecPerSec, 2) || 0 : 0,
        first_partial_ms: run ? round(run.firstPartialMs, 2) || 0 : 0,
        final_latency_ms: run ? round(run.finalLatencyMs, 2) || 0 : 0,
        roundtrip_ms: run ? round(run.roundtripMs, 2) || 0 : 0,
        wer: run ? round(run.wer, 4) || 0 : 0,
        cer: run ? round(run.cer, 4) || 0 : 0
      }
    },
    status: run ? "success" : "partial",
    artifacts: {
      raw_logs: state.logs.slice(0, 5),
      deploy_url: "https://ai-webgpu-lab.github.io/exp-voice-assistant-local/",
      runtime_adapter: describeRuntimeAdapter()
    }
  };
}

function renderStatus() {
  const badges = state.active
    ? ["Voice turn running", executionMode.label]
    : state.run
      ? [`roundtrip ${round(state.run.roundtripMs, 2)} ms`, `${round(state.run.audioSecPerSec, 2)} audioSec/s`]
      : ["Fixture ready", executionMode.label];

  elements.statusRow.innerHTML = "";
  for (const text of badges) {
    const node = document.createElement("span");
    node.className = "badge";
    node.textContent = text;
    elements.statusRow.appendChild(node);
  }

  elements.summary.textContent = state.run
    ? `Last run: first partial ${round(state.run.firstPartialMs, 2)} ms, final ${round(state.run.finalLatencyMs, 2)} ms, roundtrip ${round(state.run.roundtripMs, 2)} ms, intent ${state.run.intentId}.`
    : "Run the local voice assistant baseline to process the bundled utterance, route an intent, and stream a deterministic reply.";
}

function renderMetrics() {
  const run = state.run;
  const cards = [
    ["Audio", state.fixture ? `${state.fixture.audioSeconds} s` : "pending"],
    ["First Partial", run ? `${round(run.firstPartialMs, 2)} ms` : "pending"],
    ["Final Latency", run ? `${round(run.finalLatencyMs, 2)} ms` : "pending"],
    ["Roundtrip", run ? `${round(run.roundtripMs, 2)} ms` : "pending"],
    ["Intent", run ? run.intentId : "pending"],
    ["WER", run ? `${round(run.wer, 4)}` : "pending"]
  ];

  elements.metricGrid.innerHTML = "";
  for (const [label, value] of cards) {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<span class="label">${label}</span><div class="value">${value}</div>`;
    elements.metricGrid.appendChild(card);
  }
}

function renderEnvironment() {
  const info = [
    ["Browser", `${state.environment.browser.name} ${state.environment.browser.version}`],
    ["OS", `${state.environment.os.name} ${state.environment.os.version}`],
    ["Device", state.environment.device.class],
    ["CPU", state.environment.device.cpu],
    ["Memory", state.environment.device.memory_gb ? `${state.environment.device.memory_gb} GB` : "unknown"],
    ["Backend", state.environment.backend],
    ["Fallback", String(state.environment.fallback_triggered)],
    ["Worker Mode", state.environment.worker_mode],
    ["TTS Voice", state.run ? state.run.ttsVoice : (state.fixture ? state.fixture.ttsVoice : "pending")]
  ];

  elements.metaGrid.innerHTML = "";
  for (const [label, value] of info) {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<span class="label">${label}</span><div class="value">${value}</div>`;
    elements.metaGrid.appendChild(card);
  }
}

function renderLogs() {
  elements.logList.innerHTML = "";
  const entries = state.logs.length ? state.logs : ["No voice activity yet."];
  for (const entry of entries) {
    const li = document.createElement("li");
    li.textContent = entry;
    elements.logList.appendChild(li);
  }
}

function render() {
  renderStatus();
  renderMetrics();
  renderEnvironment();
  renderLogs();
  if (!state.transcript && !state.active && !state.run) renderTranscript("");
  if (!state.reply && !state.active && !state.run) renderReply("");
  elements.resultJson.textContent = JSON.stringify(buildResult(), null, 2);
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(buildResult(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `exp-voice-assistant-local-${state.run ? "voice-turn" : "pending"}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  log("Downloaded voice assistant readiness JSON draft.");
}

elements.runAssistant.addEventListener("click", () => {
  runAssistantTurn().catch((error) => {
    state.active = false;
    log(`Voice turn failed: ${error.message}`);
    render();
  });
});
elements.downloadJson.addEventListener("click", downloadJson);

(async function init() {
  await loadFixture();
  log("Voice assistant readiness harness ready.");
  render();
})();
