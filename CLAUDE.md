# DubShift

DubShift replaces the spoken voice in educational videos while preserving
the original video's timing and visual content.

## Initial goal

Build a local CLI application.

Eventually:

YouTube URL
→ download video
→ extract audio
→ speech-to-speech voice conversion
→ replace original audio
→ output dubbed MP4

## MVP stack

- Python 3.12+
- Typer for CLI
- yt-dlp for video downloading
- FFmpeg for media operations
- httpx for APIs
- pytest
- type hints throughout

## Architecture

Keep voice conversion behind an abstraction so multiple providers can
eventually exist.

Example:

class VoiceConverter(Protocol):
    def convert(
        self,
        input_audio: Path,
        voice_id: str,
    ) -> Path:
        ...

Possible implementations later:

- ElevenLabsVoiceConverter
- SeedVCVoiceConverter
- RVCVoiceConverter

## Rules

- Keep modules small.
- Use a src/ package layout.
- Never hardcode secrets.
- Never commit API keys.
- Never commit generated videos or audio.
- Add tests for non-network logic.
- Prefer FFmpeg and yt-dlp over reimplementing media functionality.
- Give external-command failures useful error messages.
- Do not introduce a database.
- Do not introduce authentication.
- Do not introduce Docker unless actually necessary.
- Do not build a frontend until the CLI works end-to-end.
- Do not implement features beyond the current requested milestone.

## Voice usage

DubShift should be designed around voices the user owns or has permission
to use. Do not bundle third-party character or celebrity voice models.

## Long-term CLI

Example:

dubshift "https://youtube.com/watch?v=..." \
    --provider elevenlabs \
    --voice VOICE_ID

Eventually:

dubshift "https://youtube.com/watch?v=..." \
    --provider local \
    --reference voice.wav