from cog import BasePredictor, Input, Path
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        print("Loading pipeline...")
        self.pipeline = FlaxWhisperPipline(
            "openai/whisper-large-v2",
            dtype=jnp.float16,
            batch_size=16
        )
        print("Pipeline loaded!")

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe (most formats supported)"),
        task: str = Input(
            description="Choose 'transcribe' for same-language subtitles, 'translate' for English translation",
            choices=["transcribe", "translate"],
            default="transcribe"
        ),
    ) -> dict:
        """Run a single prediction on the model"""
        try:
            # Run prediction
            result = self.pipeline(str(audio), task=task)
            
            # Return results
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            return {"error": str(e)}
