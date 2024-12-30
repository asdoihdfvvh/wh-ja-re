# Whisper-JAX Speech Recognition Model

This is a JAX implementation of OpenAI's Whisper model, optimized for faster inference. It provides up to 70x speedup compared to the original PyTorch implementation.

## Model Description

This model can:
- Transcribe speech to text in multiple languages
- Translate speech directly to English
- Process various audio formats (wav, mp3, m4a, etc.)
- Handle files of any length (automatically splits long audio)

## Input

The model accepts the following inputs:

- `audio`: An audio file (supported formats: wav, mp3, m4a, flac, etc.)
- `task`: Either "transcribe" (default) or "translate" (to English)

## Output

The model returns:
```json
{
    "text": "The transcribed or translated text",
    "language": "Detected language code",
    "segments": [
        {
            "text": "Text segment",
            "start": 0.0,
            "end": 2.5
        }
        // ... more segments
    ]
}
```

## Example Usage

```python
import replicate

# Initialize client
model = replicate.models.get("your-username/whisper-jax")

# Run prediction
output = model.predict(
    audio="path/to/audio.mp3",
    task="transcribe"
)

print(output)
```

## Performance Notes

- Uses JAX for optimized performance
- Runs on TPU/GPU for maximum speed
- Supports batch processing for improved throughput
- Uses half-precision (float16) by default for faster processing

## Limitations

- Audio files should be in a supported format
- Very noisy audio may affect transcription quality
- Performance depends on available computational resources

## Citation

```bibtex
@article{whisper,
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author = {Radford, Alec and Kim, Jong Wook and others},
  journal = {arXiv preprint arXiv:2212.04356},
  year = {2022}
}
```
