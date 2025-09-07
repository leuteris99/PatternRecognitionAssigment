## Pattern Recognition - Setup and Usage

### Prerequisites
- **Python**: 3.13 or newer (project targets `>=3.13` as per `pyproject.toml`)
- **uv**: Fast Python package manager and venv tool. Install via Homebrew on macOS:

```bash
brew install uv
```

Alternatively, see the uv docs: [uv documentation](https://docs.astral.sh/uv/).

### Create a virtual environment and install dependencies (recommended)
Use `uv sync` to create a local `.venv` and install dependencies from `pyproject.toml` (and `uv.lock` if present):

```bash
uv sync
```

This will:
- create `.venv/` in the project, and
- install all dependencies declared in `pyproject.toml`.

You do not need to activate the environment when using `uv run`, but if you prefer activation for interactive work:

```bash
source .venv/bin/activate  # macOS/Linux
```

### Alternative: manual venv + install
If you prefer to create the venv manually:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Running the code
Run the main analysis script using `uv run` so it uses your project environment [[memory:8272391]]:

```bash
uv run main.py
```

The script expects the dataset at `data/Dataset.npy` and will write outputs (plots, etc.) to the `output/` directory.

### Notes
- Plots for clustering and histograms are saved in `output/`.
- The code includes multiple clustering strategies (custom masked K-means, spherical/cosine K-means, Jaccard K-medoids).

