# TTS Server Example Calls

## Queue for playback
```bash
curl -X POST http://localhost:3500/tts -H "Content-Type: application/json" -d "{\"text\": \"Hello! The TTS server is working perfectly on your RTX 4090. Speech generation is now ready to use.\"}"
```

## Generate WAV file
```bash
curl -X POST http://localhost:3500/tts-file -H "Content-Type: application/json" -d '{"text": "Hello world, this is a test of the text to speech server"}' --output output.wav
```
