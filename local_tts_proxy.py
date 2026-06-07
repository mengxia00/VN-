import base64
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8765
MASKGCT_PYTHON_CANDIDATES = [
    Path(r"E:\ClaudeProjects\demo_01\.venv-amphion-clean\Scripts\python.exe"),
    Path(r"E:\ClaudeProjects\demo_01\.venv-amphion\Scripts\python.exe"),
]
MASKGCT_SCRIPT = Path(r"E:\ClaudeProjects\demo_01\maskgct_quick_infer.py")


def resolve_maskgct_python():
    for candidate in MASKGCT_PYTHON_CANDIDATES:
        if candidate.exists():
            return candidate
    return MASKGCT_PYTHON_CANDIDATES[0]


def _json_response(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.end_headers()
    handler.wfile.write(data)


def contains_cjk(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def synthesize_with_windows_tts(text, voice="", rate=0, volume=100):
    temp_dir = Path(tempfile.gettempdir()) / "vn_local_tts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"speech_{os.getpid()}_{abs(hash(text + voice))}.wav"

    encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
    fallback_voice = voice or ("Microsoft Huihui Desktop" if contains_cjk(text) else "Microsoft Zira Desktop")
    encoded_voice = base64.b64encode(fallback_voice.encode("utf-8")).decode("ascii")

    ps_script = f"""
Add-Type -AssemblyName System.Speech
$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_text}'))
$voiceQuery = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_voice}'))
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {int(rate)}
$synth.Volume = {int(volume)}
if ($voiceQuery) {{
  $voices = $synth.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo }}
  $match = $voices | Where-Object {{ $_.Name -like "*$voiceQuery*" -or $_.Culture.Name -like "*$voiceQuery*" }} | Select-Object -First 1
  if (-not $match) {{
    $match = $voices | Where-Object {{ $_.Culture.Name -eq 'zh-CN' }} | Select-Object -First 1
  }}
  if (-not $match) {{
    $match = $voices | Select-Object -First 1
  }}
  if ($match) {{ $synth.SelectVoice($match.Name) }}
}}
$synth.SetOutputToWaveFile('{str(output_path).replace("'", "''")}')
$synth.Speak($text)
$synth.Dispose()
"""

    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


def synthesize_with_maskgct(
    text,
    prompt_text,
    prompt_wav,
    voice="",
    prompt_language="zh",
    target_language="zh",
):
    maskgct_python = resolve_maskgct_python()
    temp_dir = Path(tempfile.gettempdir()) / "vn_local_tts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"maskgct_{os.getpid()}_{abs(hash(text + prompt_text + prompt_wav))}.wav"
    command = [
        str(maskgct_python),
        str(MASKGCT_SCRIPT),
        "--prompt-wav",
        str(prompt_wav),
        "--prompt-text",
        str(prompt_text),
        "--target-text",
        str(text),
        "--prompt-language",
        str(prompt_language),
        "--target-language",
        str(target_language),
        "--save-path",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(detail[:500]) from exc
    return output_path


class TTSHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "service": "vn-local-tts",
                    "endpoint": "/v1/audio/speech",
                    "maskgct_python": str(resolve_maskgct_python()),
                    "maskgct_ready": resolve_maskgct_python().exists() and MASKGCT_SCRIPT.exists(),
                },
            )
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/audio/speech":
            _json_response(self, 404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            text = str(payload.get("input") or payload.get("text") or "").strip()
            voice = str(payload.get("voice") or "").strip()
            model = str(payload.get("model") or "").strip().lower()
            if not text:
                _json_response(self, 400, {"error": "missing input"})
                return

            if model.startswith("maskgct"):
                prompt_text = str(payload.get("prompt_text") or "").strip()
                prompt_wav = str(payload.get("prompt_wav") or "").strip()
                if not prompt_text or not prompt_wav:
                    _json_response(
                        self,
                        400,
                        {"error": "maskgct mode requires prompt_text and prompt_wav"},
                    )
                    return
                wav_path = synthesize_with_maskgct(
                    text,
                    prompt_text=prompt_text,
                    prompt_wav=prompt_wav,
                    voice=voice,
                    prompt_language=str(payload.get("prompt_language") or "zh").strip(),
                    target_language=str(payload.get("target_language") or "zh").strip(),
                )
            else:
                wav_path = synthesize_with_windows_tts(text, voice=voice)
            data = wav_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)[:500]})

    def log_message(self, fmt, *args):
        print(f"[local-tts] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), TTSHandler)
    print(f"Local TTS proxy running: http://{HOST}:{PORT}/v1/audio/speech")
    print("Use this URL in 音效端 -> API 语音地址")
    server.serve_forever()
