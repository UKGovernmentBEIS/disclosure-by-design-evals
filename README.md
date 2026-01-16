# Disclosure by Design Evals

AI self-disclosure evaluations using text and audio inputs.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To install the dependencies, run:

```sh
uv sync
```

Create a `.env` file to store environment variables (this won't be tracked by git):

```sh
cp .env.example .env
```

To use the Azure OpenAI TTS Provider, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL` and `AZURE_OPENAI_TTS_ENDPOINT` in the `.env` file. The latter two have defaults already set. To find the `AZURE_OPENAI_API_KEY`, run the following command and copy the "SecretString" field:

```sh
aws secretsmanager get-secret-value --secret-id teams/ru/azure/tts
```

## Run Voice Evals

Run the voice evaluations using Inspect. For example, to run the tasks in `src/disclosure-by-design-evals/task_voice.py` with the dimensions (e.g. system_prompt, TTS voice, etc) specified in `config.yaml` against the `gpt-4o-audio-preview` from OpenAI, run:

```sh
uv run python src/disclosure-by-design-evals/run_voice_variants.py --model openai/gpt-4o-audio-preview --epochs 1
```

## Run Text Baseline

Run the text baseline using Inspect. For example, to run the tasks in `src/disclosure-by-design-evals/task_text.py` with the dimensions (e.g. system_prompt, etc) specified in `config.yaml` against the `gpt-4o` from OpenAI, run:

```sh
uv run python src/disclosure-by-design-evals/run_text_variants.py --model openai/gpt-4o --epochs 1
```
The outputs of these will be stored in an untracked ```log/``` folder.