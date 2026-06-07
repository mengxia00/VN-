import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(r"E:\ClaudeProjects\Amphion-main\Amphion-main")
ESPEAK_EXE = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")


def bootstrap_repo():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    # phonemizer/espeak on Windows often needs an explicit path
    if ESPEAK_EXE.exists():
        os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", str(ESPEAK_EXE))
        os.environ.setdefault("ESPEAK_NG_PATH", str(ESPEAK_EXE))
    # keep mirrors optional; if user has set HF_ENDPOINT we respect it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-wav", default=str(REPO_ROOT / "models/tts/maskgct/wav/prompt.wav"))
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--prompt-language", default="zh")
    parser.add_argument("--target-language", default="zh")
    parser.add_argument("--target-len", type=float, default=None)
    parser.add_argument("--save-path", default=str(Path(r"E:\ClaudeProjects\demo_01\outputs\maskgct_generated.wav")))
    args = parser.parse_args()

    bootstrap_repo()

    import torch
    import soundfile as sf
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from utils.util import load_config
    from models.tts.maskgct.maskgct_utils import (
        MaskGCT_Inference_Pipeline,
        build_acoustic_codec,
        build_s2a_model,
        build_semantic_codec,
        build_semantic_model,
        build_t2s_model,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[maskgct] device = {device}")

    cfg = load_config("./models/tts/maskgct/config/maskgct.json")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    downloads = {
        "semantic_codec/model.safetensors": semantic_codec,
        "acoustic_codec/model.safetensors": codec_encoder,
        "acoustic_codec/model_1.safetensors": codec_decoder,
        "t2s_model/model.safetensors": t2s_model,
        "s2a_model/s2a_model_1layer/model.safetensors": s2a_model_1layer,
        "s2a_model/s2a_model_full/model.safetensors": s2a_model_full,
    }
    for filename, model in downloads.items():
        ckpt = hf_hub_download("amphion/MaskGCT", filename=filename)
        safetensors.torch.load_model(model, ckpt)
        print(f"[maskgct] loaded {filename}")

    pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    audio = pipeline.maskgct_inference(
        args.prompt_wav,
        args.prompt_text,
        args.target_text,
        args.prompt_language,
        args.target_language,
        target_len=args.target_len,
    )
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(save_path, audio, 24000)
    print(f"[maskgct] saved to {save_path}")


if __name__ == "__main__":
    main()
