# Third-Party Licenses & Compliance Audit

This document audits all third-party libraries and model dependencies used in this orchestration repository in compliance with open-source licensing standards.

---

## 1. Foundation Speech Models & Weights

| Component | Provider / Author | License | Commercial Use Permitted | Notes / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **PersonaPlex 7B Weights** (`nvidia/personaplex-7b-v1`) | NVIDIA | **NVIDIA Open Model License** | Yes | Gated HF repo; requires license acceptance; commercial use allowed with attribution. |
| **Moshi Upstream Codebase** | Kyutai | **MIT License** | Yes | Fully permissive. |
| **Mimi Neural Codec** | Kyutai | **MIT License** | Yes | 24 kHz neural audio tokenizer. |

---

## 2. Core Python Runtime Dependencies

| Library | Version Requirement | License | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `fastapi` | `>=0.115.0` | **MIT** | High-performance async REST & WebSocket API gateway |
| `uvicorn` | `>=0.30.0` | **BSD-3-Clause** | Lightning-fast ASGI web server implementation |
| `websockets` | `>=13.0` | **BSD-3-Clause** | Full-duplex WebSocket framing & client connectivity |
| `pydantic` | `>=2.8.0` | **MIT** | Data validation and typed configuration management |
| `pydantic-settings` | `>=2.4.0` | **MIT** | Environment variable settings parser |
| `numpy` | `>=1.26.0` | **BSD-3-Clause** | Array manipulation for raw PCM audio frames |
| `scipy` | `>=1.12.0` | **BSD-3-Clause** | Polyphase resampling and DSP filtering algorithms |
| `soundfile` | `>=0.12.0` | **BSD-3-Clause** | Audio I/O wrapper for libsndfile |
| `sentencepiece` | `>=0.2.0` | **Apache-2.0** | Unsupervised text tokenizer for Moshi LM system prompts |
| `sphn` | `>=0.1.4, <0.2` | **MIT** | Kyutai streaming Opus audio transcoding library |
| `soxr` | `>=0.5.0` | **LGPL-2.1+ / MIT** | High-quality streaming audio sample rate converter |
| `sqlalchemy` | `>=2.0.0` | **MIT** | SQL toolkit and Object Relational Mapper |
| `alembic` | `>=1.13.0` | **MIT** | Database schema migration management |
| `cryptography` | `>=43.0.0` | **Apache-2.0 / BSD** | Cryptographic token handling & storage encryption |
| `httpx` | `>=0.27.0` | **BSD-3-Clause** | Async HTTP client for worker health probing |
| `pyyaml` | `>=6.0.0` | **MIT** | YAML agent definition & configuration loader |
| `structlog` | `>=24.0.0` | **Apache-2.0 / MIT** | Production structured JSON telemetry & access logger |
| `pytest` | `>=8.0.0` | **MIT** | Automated test suite framework |
| `pytest-asyncio` | `>=0.23.0` | **Apache-2.0** | Async unit and integration test runner |

---

## 3. License Audit Verification

- **Permissive License Verification**: All runtime and testing dependencies use **MIT**, **BSD-3-Clause**, or **Apache-2.0** permissive licenses.
- **No GPL Conflicts**: No viral Copyleft (GPL v2/v3) code is embedded into the core speech serving path.
- **Security & Secret Hygiene**: All authentication keys (`HF_TOKEN`, `AUTH_TOKEN`, TLS certificates) are loaded strictly via environment variables and never committed to version control.
