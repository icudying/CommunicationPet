# Electron desktop frontend

This folder contains the Electron shell for CommunicationPet.

- Custom frameless window, voice orb, persona switcher and settings/log panels
- Starts and supervises the Python realtime backend (WebSocket `ws://127.0.0.1:8011/ws/realtime`)
- Streams microphone audio, transcripts and assistant audio
- Optional `<webview>` bridge for the avatar DLC

## Run

```bash
npm install
npm run dev
```

The main process resolves Python in this order: `MINIMIND_PYTHON`, a bundled
`runtime/python`, `CONDA_PREFIX`, `PYTHON`, then `python` on `PATH`.
