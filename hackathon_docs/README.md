Welcome to Modulate's Velma-2 API!  Velma-2 is a voice transcription and understanding model - so if you have any voice or audio that you want to process or understand, these APIs are for you!

## Endpoints:

* https://modulate-prototype-apis.com/api/velma-2-stt-streaming : A websocket-based voice transcription API, with features for speaker diarization, emotion detection, accent detection, PHI/PII tagging, and more!
* https://modulate-prototype-apis.com/api/velma-2-stt-batch : A REST async voice transcription API - feature parity with streaming, but higher throughput for pre-recorded files
* https://modulate-prototype-apis.com/api/velma-2-stt-batch-english-vfast : A REST async voice transcription API - trading off everything in favor of high speed and throughput.  Lower-level API for english-only and opus format audio.  Most appropriate for high scale batch audio processing


## Getting Started:

* Each API has an associated .yaml file in this directory with an OpenAPI spec for the endpoint.  There are also python test files for running audio through the APIs, which can be used as a reference.
* APIs have 100 hours of free usage included - reach out to Carter on the hackathon discord - username `carterhuffman8385` - for an API key!
* Please reach out to `carterhuffman8385` in Discord, or in the #modulate-ai channel, if you have any questions!