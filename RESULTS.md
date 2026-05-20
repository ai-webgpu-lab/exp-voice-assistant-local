# Results

## 1. 실험 요약
- 저장소: exp-voice-assistant-local
- 커밋 해시: e4f02bd
- 실험 일시: 2026-05-20T15:46:38.971Z -> 2026-05-20T15:46:41.409Z
- 담당자: ai-webgpu-lab
- 실험 유형: `audio`
- 상태: `success`

## 2. 질문
- 로컬 voice assistant 실험으로 넘기기 전에 STT partial, final latency, roundtrip 보고 경로를 먼저 고정할 수 있는가
- wake word, intent route, TTS voice, fallback metadata가 audio 결과 문서에 같이 남는가
- 실제 STT, local planner, TTS runtime 교체 전 deterministic voice-turn harness로 반복 검증이 가능한가

## 3. 실행 환경
### 브라우저
- 이름: Chrome
- 버전: 147.0.7727.15

### 운영체제
- OS: Linux
- 버전: unknown

### 디바이스
- 장치명: Linux x86_64
- device class: `desktop-high`
- CPU: 16 threads
- 메모리: 32 GB
- 전원 상태: `unknown`

### GPU / 실행 모드
- adapter: navigator.gpu available
- backend: `webgpu`
- fallback triggered: `false`
- worker mode: `worker`
- cache state: `warm`
- required features: ["shader-f16"]
- limits snapshot: {"maxBindGroups":4}

## 4. 워크로드 정의
- 시나리오 이름: Voice Assistant Local Readiness
- 입력 프로필: 6.8s-2-segments
- 데이터 크기: wakeWord=aurora; segments=2; intent=lab_status_brief; ttsVoice=alloy-local; firstAudioMs=194.4; roundtripMs=267.2; backend=webgpu; fallback=false; automation=playwright-chromium, wakeWord=aurora; segments=2; intent=lab_status_brief; ttsVoice=alloy-local; firstAudioMs=196.5; roundtripMs=269.6; backend=webgpu; fallback=false; realAdapter=fallback(adapter.loadModel is not a function); automation=playwright-chromium
- dataset: voice-fixture-v1
- model_id 또는 renderer: deterministic-voice-assistant-local-v1
- 양자화/정밀도: -
- resolution: -
- context_tokens: -
- output_tokens: -

## 5. 측정 지표
### 공통
- time_to_interactive_ms: 419.3 ~ 1271.2 ms
- init_ms: 267.2 ~ 269.6 ms
- success_rate: 1
- peak_memory_note: 32 GB reported by browser
- error_type: -

### STT / Voice
- audio_sec_per_sec: 89.47 ~ 90.91
- first_partial_ms: 34.1 ~ 34.3 ms
- final_latency_ms: 74.8 ~ 76 ms
- roundtrip_ms: 267.2 ~ 269.6 ms
- wer: 0
- cer: 0
- worker modes: worker
- backends: webgpu
- fallback states: false

## 6. 결과 표
| Run | Scenario | Backend | Cache | Mean | P95 | Notes |
|---|---|---:|---:|---:|---:|---|
| 1 | Voice Assistant Local Readiness | webgpu | warm | 90.91 | 267.2 | first_partial=34.3 ms, final=74.8 ms, WER=0 |
| 2 | Voice Assistant Local Readiness | webgpu | warm | 89.47 | 269.6 | first_partial=34.1 ms, final=76 ms, WER=0 |

## 7. 관찰
- local voice assistant readiness baseline은 backend=webgpu, fallback_triggered=false, worker_mode=worker로 기록됐다.
- voice summary는 first_partial_ms=34.3, final_latency_ms=74.8, roundtrip_ms=267.2였다.
- voice assistant metadata는 wakeWord=aurora; segments=2; intent=lab_status_brief; ttsVoice=alloy-local; firstAudioMs=194.4; roundtripMs=267.2; backend=webgpu; fallback=false; automation=playwright-chromium로 남았다.
- playwright-chromium로 수집된 automation baseline이며 headless=true, browser=Chromium 147.0.7727.15.
- 실제 runtime/model/renderer 교체 전 deterministic harness 결과이므로, 절대 성능보다 보고 경로와 재현성 확인에 우선 의미가 있다.

## 8. Real Adapter vs Deterministic
- adapter: real=voice-assistant-xenova-whisper-tiny-xenova-phi-3-mini-4k-instruct-q4f16-300, deterministic=deterministic-mock
- adapter_run: real=connected, deterministic=deterministic
- success_rate: real=1, deterministic=1

## 9. 결론
- local voice assistant readiness harness가 STT, intent routing, reply draft, TTS roundtrip 결과를 같은 문서에 남기게 됐다.
- 다음 단계는 deterministic voice turn을 실제 STT runtime, local planner, TTS provider로 교체하되 first_partial/final_latency/roundtrip metric 구조를 유지하는 것이다.
- 이후 `bench-voice-roundtrip`과 `app-voice-agent-lab`의 roundtrip regression 기준으로 재사용할 수 있다.

## 10. 첨부
- 스크린샷: ./reports/screenshots/01-voice-assistant-local-readiness.png, ./reports/screenshots/10-voice-assistant-local-real-voice-assistant.png
- 로그 파일: ./reports/logs/01-voice-assistant-local-readiness.log, ./reports/logs/10-voice-assistant-local-real-voice-assistant.log
- raw json: ./reports/raw/01-voice-assistant-local-readiness.json, ./reports/raw/10-voice-assistant-local-real-voice-assistant.json
- 배포 URL: https://ai-webgpu-lab.github.io/exp-voice-assistant-local/
- 관련 이슈/PR: -
