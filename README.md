# Disclosure by Design Evals

AI self-disclosure evaluations using text and audio inputs.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To install the dependencies, run:

```sh
uv sync
```

## API Keys & Configuration

### Required API Keys

This project requires API keys from AI providers. You will need API keys for any model you want to evaluate. In addition you will need API keys for additional functionality. For example:

**OpenAI API Key** GPT-4o is used for scoring (alternative models are possible but not validated).
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Create an API key at [API Keys page](https://platform.openai.com/api-keys)
   - Set as `OPENAI_API_KEY` in your `.env` file

2. **OpenAI TTS** - Only needed if using TTS for speech evaluations
   - Requires: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_TTS_ENDPOINT`



### Setup Steps

1. Copy the example environment file:
   ```sh
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```sh
   # Open in your preferred editor
   nano .env
   ```

3. At minimum, set your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-proj-...your-actual-key...
   ```

4. The `.env` file is gitignored and will never be committed to version control.

### API Key Sources

The project uses [inspect-ai](https://inspect.ai-safety-institute.org.uk/) which automatically reads API keys from:
- Environment variables in `.env` file (recommended)
- System environment variables
- Standard locations like `~/.openai/api_key`

**Note:** This repo previously used AISI-internal tools for key management. Those dependencies have been removed for open-source release.

## Evaluate AI self-disclosure in speech interactions

Run the voice evaluations using Inspect. For example, to run the tasks in `src/disclosure-by-design-evals/task_voice.py` with the dimensions (e.g. system_prompt, TTS voice, etc) specified in `config.yaml` against the `gpt-4o-audio-preview` from OpenAI, run:

```sh
uv run python src/disclosure-by-design-evals/run_voice_variants.py --model openai/gpt-4o-audio-preview --epochs 1
```

## Evaluate AI self-disclosure in text interactions

Run the text baseline using Inspect. For example, to run the tasks in `src/disclosure-by-design-evals/task_text.py` with the dimensions (e.g. system_prompt, etc) specified in `config.yaml` against the `gpt-4o` from OpenAI, run:

```sh
uv run python src/disclosure-by-design-evals/run_text_variants.py --model openai/gpt-4o --epochs 1
```
The outputs of these will be stored in an untracked ```log/``` folder.

## Considerations
### Troubleshooting API Keys

**"No API key found" errors:**
- Ensure your `.env` file exists in the project root
- Check that `OPENAI_API_KEY` is set (not just a placeholder)
- Try exporting the key directly: `export OPENAI_API_KEY=sk-...`

**Rate limit errors:**
- OpenAI API has rate limits based on your account tier
- Consider adding delays between requests or using a higher-tier account

**Model not found errors:**
- Verify your API key has access to the models you're trying to use
- Some models (like gpt-4o-audio-preview) may require waitlist access

### Cost Estimation

**⚠️ Important:** Running these evaluations will consume API credits.

Approximate costs per run (with default settings):
- **Text baseline**: ~$X per epoch (X prompts × model calls)
- **Voice evaluation**: ~$Y per epoch (includes TTS generation + audio inference + scoring)

Tips to minimize costs during testing:
- Start with `--epochs 1` 
- Test with a single system message variant first
- Use cheaper models for initial testing (gpt-3.5-turbo for scoring)