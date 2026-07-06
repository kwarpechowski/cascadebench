# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Optional LLM configuration for persona (twin) regeneration.

The CascadeBench BENCHMARK does not need this. The graphs, cascade attacks, detectors,
and evaluation are fully procedural, and the synthetic population is shipped pre-generated
in ``data/twins/*.json``. You only need an LLM to *regenerate* new C1..C8 personas with
``data/twin_generator.py``.

To enable that path: copy this file to ``config.py`` (kept out of version control) and set
the environment variables below. ``config.py`` must expose ``PPD_TWIN_MODEL`` and
``make_client_for(model)`` returning an OpenAI-compatible client.

    export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your internal endpoint
    export OPENAI_API_KEY="sk-..."
    export PPD_TWIN_MODEL="gpt-4o-mini"                  # any OpenAI-compatible chat model
    pip install -e .[llm]
"""
import os

# Chat model used to fabricate personas (temperature-0 -> deterministic personas).
PPD_TWIN_MODEL = os.environ.get("PPD_TWIN_MODEL", "gpt-4o-mini")


def make_client_for(model: str):
    """Return an OpenAI-compatible client (requires the ``openai`` package).

    Reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment. If your endpoint is
    served behind a private/corporate certificate authority, inject the OS trust store
    first (``pip install truststore``):

        import truststore; truststore.inject_into_ssl()
    """
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> OpenAI public default
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(base_url=base_url, api_key=api_key)
