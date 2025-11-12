# PIE-Airbus-Helicopter

## Setup

### Python packages

Using `uv` is highly recommendable once this project was created using it. To
install `uv` from `pip` is really straight forward:

```bash
pip install uv
```

Alternatively, you can look into the 
[uv documentation](https://docs.astral.sh/uv/getting-started/installation/)
to see how to install it through a standalone installer for your OS.

Then, to install all dependencies one must simply type:

```bash
uv sync
```

If you don't want to use uv, you can still download the required dependencies 
with your usual python package manager (pip, poetry, conda ...). For this, 
refer to the `dependencies` part of the `pyproject.toml` file to see what 
packages need to be downloaded.