#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

_LOGGER = logging.getLogger("piper_train.infer")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog="piper_train.infer")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument("--output-dir", required=True, help="Path to write WAV files")
    parser.add_argument("--sample-rate", type=int, default=22050)
    #
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.8)
    #
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info("Using device: %s", device)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)
    model.to(device)  # Move weights to GPU

    # Inference only
    model.eval()

    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue

        utt = json.loads(line)
        utt_id = str(i)
        phoneme_ids = utt["phoneme_ids"]
        speaker_id = utt.get("speaker_id")

        text = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)
        text_lengths = torch.LongTensor([len(phoneme_ids)]).to(device)
        scales = [args.noise_scale, args.length_scale, args.noise_w]
        sid = torch.LongTensor([speaker_id]).to(device) if speaker_id is not None else None

        scales = [args.noise_scale, args.length_scale, args.noise_w]

        start_time = time.perf_counter()

        with torch.no_grad():
            audio = model(text, text_lengths, scales, sid=sid)
            audio = audio.detach().cpu().numpy() # Bring back to CPU here

        audio = audio_float_to_int16(audio)
        end_time = time.perf_counter()

        audio_duration_sec = audio.shape[-1] / args.sample_rate
        infer_sec = end_time - start_time
        real_time_factor = (
            infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        )

        _LOGGER.debug(
            "Real-time factor for %s: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
            i + 1,
            real_time_factor,
            infer_sec,
            audio_duration_sec,
        )

        output_path = args.output_dir / f"{utt_id}.wav"
        write_wav(str(output_path), args.sample_rate, audio)


if __name__ == "__main__":
    main()
