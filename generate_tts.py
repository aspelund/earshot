"""
Chatterbox-Turbo TTS Generation Script
Generate speech from text using Resemble AI's Chatterbox-Turbo model.
"""
import re
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
from pathlib import Path


def split_into_chunks(text: str, max_sentences: int = 3, max_length: int = 250) -> list[str]:
    """Split text into chunks with up to max_sentences or max_length characters."""
    # Split on sentence-ending punctuation, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        # Start new chunk if adding this sentence exceeds limits
        if current_chunk and (
            len(current_chunk) >= max_sentences
            or current_length + sentence_len > max_length
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    # Read input text
    input_file = Path("input.txt")
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return

    text = input_file.read_text()
    chunks = split_into_chunks(text)
    print(f"Loaded {len(chunks)} chunks from {input_file}")

    print("\nLoading Chatterbox-Turbo model...")

    # Check for CUDA availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ChatterboxTurboTTS.from_pretrained(device=device)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Check for reference audio (optional, for voice cloning)
    ref_audio = Path("imj.mp3")
    audio_prompt = str(ref_audio) if ref_audio.exists() else None

    if audio_prompt:
        print(f"Using reference audio: {ref_audio}")
    else:
        print("No reference audio found. Generating with default voice.")

    # Generate each chunk
    for i, text in enumerate(chunks, 1):
        print(f"\n[{i}/{len(chunks)}] Generating: {text[:60]}...")

        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt,
        )

        output_path = output_dir / f"chunk_{i:02d}.wav"
        ta.save(str(output_path), wav, model.sr)
        print(f"  Saved to: {output_path}")

    print(f"\nDone! Generated {len(chunks)} audio files in '{output_dir}/'")


if __name__ == "__main__":
    main()
