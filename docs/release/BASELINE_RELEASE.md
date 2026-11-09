> Baseline preview generated against Feifan f488f9ff with uncommitted US6.2 additions. NOT deployment approval. Regenerate from the final clean release commit.

# Release 0.7.0-us6.2-preview

Source commit: `f488f9ff18b459fff267d3e080a0e34fc769aeb4`

Working tree dirty: True

## Models

```json
{
  "brain_openai_configured": "gpt-4o-mini",
  "zoe_chat_hardcoded": "gpt-4o-mini",
  "faq_chat_hardcoded": "gpt-4o-mini",
  "judge_default": "gpt-4o-mini",
  "zoe_tts": "tts-1-hd",
  "vector_default": "text-embedding-3-small",
  "faq_embedding": "SDK default; resolve from captured langchain-openai version"
}
```

## Prompt versions

Whole source files are versioned conservatively; non-prompt edits may also change a version.

- `backend/agents/zoe/__init__.py`: `sha256:cdb58c1708653b32968052d4470e66c3f85adc527a0afdf5922650779dbbc760`
- `backend/agents/zoe/helpers.py`: `sha256:69ac32d5745c5714359f31a3ab03c00f57c797b59790a4fa045d797284ef3505`
- `backend/agents/zoe/personality.py`: `sha256:fa3c530788b1ab15f0e4d3b36aeb0547c07d15c3ab553585b16b910d9e356a40`
- `backend/agents/zoe/zoe_core.py`: `sha256:c1d03a15963a8fbfe40a67c021d945c84757970936c63e77f772d4e21e30889f`
- `backend/brain/cortex/reasoning_engine.py`: `sha256:d4878d06a70d0dc6f3f98d824abb91d8a086b2383fcbd4d9275f6a3c21243922`
- `backend/brain/cortex/workflow_engine.py`: `sha256:773772b28e1d6ea12eeaa6311e1935a266cf1c90a96fe7d4180fac823793773b`
- `backend/brain/guardrails/fallback_handler.py`: `sha256:1002aa874ec892558b43c5796bb2dd8b0de25b47a084ebb51149ff74e6e194cf`
- `backend/brain/guardrails/policy_validator.py`: `sha256:c0083ba9f49666d1757d051b4c81d9b0e51a51c30160a95634da2796a3d48310`
- `backend/brain/providers/gemini.py`: `sha256:fac6462febd26326fd858e61f0e5f40bada93ffa08d2168d282f0ab94d88398d`
- `backend/brain/providers/openai.py`: `sha256:53444c41ee455ec8d06fbb9883e26afbff7cb547922b0239d274f51e067c0c75`
- `backend/evaluators/__init__.py`: `sha256:ff6fa97baedfa4be673c95f2c7bca968815398ea7fdca801977d31fc281889dd`
- `backend/evaluators/base.py`: `sha256:3a33b22e59e734c6eae2a37377ff561b673c5dc6e55521456b67f34ebb92fac8`
- `backend/evaluators/common_evaluators/empathy_tone_evaluator.py`: `sha256:8450bfb91f8ee06cab354cf3e094e7bdbe81bd61172b314f358a7fbcdcef6c56`
- `backend/evaluators/common_evaluators/factuality_hallucination_risk_evaluator.py`: `sha256:026b2bb05354a1de62a59ddcabf27d9799ab652f4c840554961948a96848bbac`
- `backend/evaluators/common_evaluators/instruction_following_relevance_evaluator.py`: `sha256:b00da0cb984fbec0f58259dc66fa81668c495099f547226a13d7cfe79e31f73e`
- `backend/evaluators/common_evaluators/safety_harm_evaluator.py`: `sha256:1b45778341ac0c90d6f383c4d3f252579346e729caf0280c908d8ee74358fff7`
- `backend/evaluators/faq_evaluators/abstention_evaluator.py`: `sha256:226b354d1b8a5c6fcb9581b57da16372acf446621b4702676c62fb2a15072e43`
- `backend/evaluators/faq_evaluators/answer_faithfulness_evaluator.py`: `sha256:f352406cbfd89189b5214bcbbf533ddffa391c664898860f0cd3af145d61eaab`
- `backend/evaluators/faq_evaluators/citation_evaluator.py`: `sha256:41ebda6228c8dd3043820c1f590afe5dbba93f37e8fc14765beb07a363f78020`
- `backend/evaluators/faq_evaluators/faq_exactness_evaluator.py`: `sha256:b398fddf1b4c80428db5b1c692569d30f7d9d52fb860d84e11bec6b0d7b14f72`
- `backend/evaluators/faq_evaluators/policy_consistency_evaluator.py`: `sha256:c8264e6d97d3aae9eb623e9e48f63f538ff8a94fd603bf9e0de758fd3b60c429`
- `backend/plugins/zoe_agent.py`: `sha256:13ebb1e37a023e2d221a29a836ef56680706089550911e7a6272aceda1505005`
- `faq-bot-rag/app/rag_chat.py`: `sha256:9cde6ef1a98a7ed93938311de0f0822bbca076962975c6fdbf313caee31e95fe`

## Dependencies

Python declarations (not installed-version locks):

### requirements.txt

```text
fastapi
uvicorn
pydantic
python-dotenv
requests
beautifulsoup4
pymupdf
langchain
langchain-community
langchain-openai
faiss-cpu
tiktoken
```

### backend/requirements.txt

```text
fastapi
uvicorn[standard]
pydantic
python-dotenv
openai
langchain
langchain-openai
langchain-chroma
langgraph
chromadb
python-multipart
anthropic
httpx
langfuse>=3.7.0
google-genai
tavily-python
PyJWT
cryptography
requests
urllib3<2.0.0
importlib-metadata>=6.0.0
```

### faq-bot-rag/requirements.txt

```text
fastapi
uvicorn
pydantic
python-dotenv
requests
beautifulsoup4
pymupdf
langchain
langchain-community
langchain-openai
faiss-cpu
tiktoken
```

### Frontend

| Package | Declared | Lockfile root resolution |
| --- | --- | --- |
| @hookform/resolvers | ^3.9.1 | 3.10.0 |
| @radix-ui/react-accordion | 1.2.2 | 1.2.2 |
| @radix-ui/react-alert-dialog | 1.1.4 | 1.1.4 |
| @radix-ui/react-aspect-ratio | 1.1.1 | 1.1.1 |
| @radix-ui/react-avatar | 1.1.2 | 1.1.2 |
| @radix-ui/react-checkbox | 1.1.3 | 1.1.3 |
| @radix-ui/react-collapsible | 1.1.2 | 1.1.2 |
| @radix-ui/react-context-menu | 2.2.4 | 2.2.4 |
| @radix-ui/react-dialog | 1.1.4 | 1.1.4 |
| @radix-ui/react-dropdown-menu | 2.1.4 | 2.1.4 |
| @radix-ui/react-hover-card | 1.1.4 | 1.1.4 |
| @radix-ui/react-label | 2.1.1 | 2.1.1 |
| @radix-ui/react-menubar | 1.1.4 | 1.1.4 |
| @radix-ui/react-navigation-menu | 1.2.3 | 1.2.3 |
| @radix-ui/react-popover | 1.1.4 | 1.1.4 |
| @radix-ui/react-progress | 1.1.1 | 1.1.1 |
| @radix-ui/react-radio-group | 1.2.2 | 1.2.2 |
| @radix-ui/react-scroll-area | 1.2.2 | 1.2.2 |
| @radix-ui/react-select | 2.1.4 | 2.1.4 |
| @radix-ui/react-separator | 1.1.1 | 1.1.1 |
| @radix-ui/react-slider | 1.2.2 | 1.2.2 |
| @radix-ui/react-slot | 1.1.1 | 1.1.1 |
| @radix-ui/react-switch | 1.1.2 | 1.1.2 |
| @radix-ui/react-tabs | 1.1.2 | 1.1.2 |
| @radix-ui/react-toast | 1.2.4 | 1.2.4 |
| @radix-ui/react-toggle | 1.1.1 | 1.1.1 |
| @radix-ui/react-toggle-group | 1.1.1 | 1.1.1 |
| @radix-ui/react-tooltip | 1.1.6 | 1.1.6 |
| @types/bcryptjs | ^2.4.6 | 2.4.6 |
| bcryptjs | ^3.0.2 | 3.0.3 |
| class-variance-authority | ^0.7.1 | 0.7.1 |
| clsx | ^2.1.1 | 2.1.1 |
| cmdk | 1.0.4 | 1.0.4 |
| date-fns | ^3.6.0 | 3.6.0 |
| embla-carousel-react | 8.5.1 | 8.5.1 |
| framer-motion | ^10.18.0 | 10.18.0 |
| input-otp | 1.4.1 | 1.4.1 |
| keycloak-js | ^26.0.0 | 26.2.1 |
| lucide-react | ^0.454.0 | 0.454.0 |
| next | ^16.0.10 | 16.0.10 |
| next-themes | ^0.4.4 | 0.4.6 |
| react | 18.3.1 | 18.3.1 |
| react-day-picker | ^9.1.3 | 9.11.3 |
| react-dom | 18.3.1 | 18.3.1 |
| react-hook-form | ^7.54.1 | 7.68.0 |
| react-is | ^18.3.1 | 18.3.1 |
| react-resizable-panels | ^2.1.7 | 2.1.9 |
| recharts | 2.15.0 | 2.15.0 |
| resend | ^4.5.2 | 4.8.0 |
| sonner | ^1.7.1 | 1.7.4 |
| styled-components | ^6.1.19 | 6.1.19 |
| tailwind-merge | ^2.5.5 | 2.6.0 |
| tailwindcss-animate | ^1.0.7 | 1.0.7 |
| vaul | ^1.1.2 | 1.1.2 |
| zod | ^4.0.14 | 4.1.13 |
| @types/node | ^22 | 22.19.1 |
| @types/react | ^18.3.12 | 18.3.27 |
| @types/react-dom | ^18.3.1 | 18.3.7 |
| @types/styled-components | ^5.1.34 | 5.1.36 |
| autoprefixer | ^10.4.21 | 10.4.22 |
| postcss | ^8.5.6 | 8.5.6 |
| tailwindcss | ^3.4.17 | 3.4.18 |
| typescript | ^5 | 5.9.3 |

### Containers

- `quay.io/keycloak/keycloak:24.0.1` (tag; record deployed digest)
- `postgres:15` (tag; record deployed digest)

## Limitations

- Model aliases do not pin a provider snapshot.
- Source references include optional providers and defaults, not only active models.
- Python requirements and container tags are not immutable locks; archive installed versions and image digests.
- No backend commit endpoint exists; verify backend artifact digest in deployment evidence.

Complete transitive frontend versions and file hashes: manifest.json. Deployment and rollback sign-off: docs/release/RELEASE_RECORD.md.
