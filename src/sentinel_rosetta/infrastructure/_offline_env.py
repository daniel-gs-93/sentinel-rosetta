"""Resource-aware: sets offline environment variables; import it first."""

import os

# Imported for its side effect. Must run before transformers/huggingface_hub
# are imported, since they read these as constants at import time; setting
# them later would be too late to stop AutoTokenizer from silently making a
# network call when a local model path is missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
