# Benchmarking Examples

This directory contains example scripts demonstrating how to use the OpenRouter VLM benchmarking module.

## Available Examples

### 1. `demo_structure.py` - Module Structure Demo

Shows the structure of the benchmarking module, available task sets, and expected output formats without making API calls.

**Run:**

```bash
uv run python examples/demo_structure.py
```

**What it demonstrates:**

- All available task sets (WCAG, simple, detailed)
- CLI command examples
- Python API usage
- Expected results structure

### 2. `benchmark_example.py` - Full Benchmarking Example

Complete example showing how to use the benchmarking API programmatically. Requires an OpenRouter API key.

**Setup:**

```bash
# Copy example.env to .env and add your API key
cp example.env .env
# Edit .env to add OPENROUTER_API_KEY
```

**Run:**

```bash
uv run python examples/benchmark_example.py
```

**What it demonstrates:**

- Listing available models
- Benchmarking single models
- Comparing multiple models
- Working with different task sets
- Error handling

## Quick Start

1. **Explore the module structure** (no API key needed):

   ```bash
   uv run python examples/demo_structure.py
   ```

2. **Set up your environment** for actual benchmarking:

   ```bash
   cp example.env .env
   # Add your OPENROUTER_API_KEY to .env
   uv sync
   ```

3. **Try the full example**:
   ```bash
   uv run python examples/benchmark_example.py
   ```

## Using the CLI Directly

For actual benchmarking, use the CLI commands:

```bash
# List available models
uv run python -m src.benchmarking list-models

# Benchmark a single model
uv run python -m src.benchmarking benchmark --model "openai/gpt-4-vision-preview" --task-set wcag

# Compare multiple models
uv run python -m src.benchmarking benchmark-multiple --models "model1,model2" --output results.json
```

## More Information

For complete documentation, see:

- [Benchmarking Module README](https://github.com/maehr/chr2025-seeing-history-unseen/blob/main/src/benchmarking/README.md)
- [Main Repository README](https://github.com/maehr/chr2025-seeing-history-unseen/blob/main/README.md)
