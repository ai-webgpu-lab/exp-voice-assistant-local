// Real local voice assistant runtime sketch for exp-voice-assistant-local.
//
// Gated by ?mode=real-voice-assistant. Default deterministic harness path is
// untouched. The adapter loads two pipelines (Whisper STT + Phi-3 reply) and
// exposes prefill/decode that drive STT → intent → reply in one path.
// `loadVoiceFromCdn` is parameterized so tests can inject a stub.

const DEFAULT_TRANSFORMERS_VERSION = "3.0.0";
const DEFAULT_TRANSFORMERS_CDN = (version) => `https://esm.sh/@huggingface/transformers@${version}`;
const DEFAULT_STT_ID = "Xenova/whisper-tiny";
const DEFAULT_REPLY_ID = "Xenova/Phi-3-mini-4k-instruct-q4f16";

export async function loadVoiceFromCdn({ version = DEFAULT_TRANSFORMERS_VERSION } = {}) {
  const transformers = await import(/* @vite-ignore */ DEFAULT_TRANSFORMERS_CDN(version));
  if (!transformers || typeof transformers.pipeline !== "function") {
    throw new Error("transformers module did not expose pipeline()");
  }
  return { transformers, pipeline: transformers.pipeline, env: transformers.env };
}

export function buildRealVoiceAssistantAdapter({
  pipeline,
  env,
  version = DEFAULT_TRANSFORMERS_VERSION,
  sttModelId = DEFAULT_STT_ID,
  replyModelId = DEFAULT_REPLY_ID
}) {
  if (typeof pipeline !== "function") {
    throw new Error("buildRealVoiceAssistantAdapter requires a callable pipeline");
  }
  const slug = `${sttModelId}-${replyModelId}`.replace(/[^A-Za-z0-9]/g, "-").toLowerCase();
  const id = `voice-assistant-${slug}-${version.replace(/[^0-9]/g, "")}`;
  let stt = null;
  let reply = null;

  return {
    id,
    label: `Voice Assistant (Transformers.js ${version})`,
    version,
    capabilities: ["prefill", "decode", "asr", "reply", "fixed-output-budget"],
    loadType: "async",
    backendHint: "webgpu",
    isReal: true,
    async loadRuntime({ device = "webgpu", sttDtype = "q4", replyDtype = "q4" } = {}) {
      if (env && typeof env === "object") env.allowRemoteModels = true;
      stt = await pipeline("automatic-speech-recognition", sttModelId, { device, dtype: sttDtype });
      reply = await pipeline("text-generation", replyModelId, { device, dtype: replyDtype });
      return { stt, reply };
    },
    async prefill(_runtime, prompt) {
      const startedAt = performance.now();
      const audio = (prompt && prompt.audio) || null;
      const sampleCount = audio && (audio.length || audio.byteLength) || 0;
      const intent = (prompt && prompt.intent) || "general";
      const prefillMs = performance.now() - startedAt;
      return { promptTokens: sampleCount, prefillMs, audio, intent };
    },
    async decode(activeRuntime, prefillResult, outputTokenBudget = 64) {
      const sttTarget = (activeRuntime && activeRuntime.stt) || stt;
      const replyTarget = (activeRuntime && activeRuntime.reply) || reply;
      if (!sttTarget || !replyTarget) {
        throw new Error("real voice-assistant adapter requires loadRuntime() before decode()");
      }
      const sttStart = performance.now();
      const transcript = await sttTarget(prefillResult.audio || "audio-fixture", { return_timestamps: false });
      const sttMs = performance.now() - sttStart;
      const transcriptText = transcript && transcript.text ? transcript.text : "";
      const replyStart = performance.now();
      const promptText = `Intent: ${prefillResult.intent}\nUser: ${transcriptText}\nAssistant:`;
      const output = await replyTarget(promptText, { max_new_tokens: outputTokenBudget, return_full_text: false });
      const replyMs = performance.now() - replyStart;
      const replyText = Array.isArray(output) && output[0] && output[0].generated_text ? output[0].generated_text : "";
      const decodeMs = sttMs + replyMs;
      const tokens = replyText.split(/\s+/).filter(Boolean).length || outputTokenBudget;
      return {
        tokens,
        decodeMs,
        transcript: transcriptText,
        text: replyText,
        sttMs,
        replyMs,
        ttftMs: decodeMs / Math.max(tokens, 1),
        decodeTokPerSec: tokens / Math.max(decodeMs / 1000, 0.001)
      };
    }
  };
}

export async function connectRealVoiceAssistant({
  registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null,
  loader = loadVoiceFromCdn,
  version = DEFAULT_TRANSFORMERS_VERSION,
  sttModelId = DEFAULT_STT_ID,
  replyModelId = DEFAULT_REPLY_ID
} = {}) {
  if (!registry) {
    throw new Error("runtime registry not available");
  }
  const { pipeline, env } = await loader({ version });
  if (typeof pipeline !== "function") {
    throw new Error("loaded pipeline is not callable");
  }
  const adapter = buildRealVoiceAssistantAdapter({ pipeline, env, version, sttModelId, replyModelId });
  registry.register(adapter);
  return { adapter, pipeline, env };
}

if (typeof window !== "undefined" && window.location && typeof window.location.search === "string") {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "real-voice-assistant" && !window.__aiWebGpuLabRealVoiceAssistantBootstrapping) {
    window.__aiWebGpuLabRealVoiceAssistantBootstrapping = true;
    connectRealVoiceAssistant().catch((error) => {
      console.warn(`[real-voice-assistant] bootstrap failed: ${error.message}`);
      window.__aiWebGpuLabRealVoiceAssistantBootstrapError = error.message;
    });
  }
}
