FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONNOUSERSITE=1 \
    PTCG_EVAL_RUNTIME_ID=ptcg-release-eval-py311-torch260cpu-v1

RUN python -m pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.6.0" \
    && python -m pip install \
      "hydra-core==1.3.2" \
      "loguru==0.7.3" \
      "numpy==2.2.0" \
      "omegaconf==2.3.0" \
      "orjson==3.10.15" \
      "pyarrow==19.0.1" \
      "pydantic==2.11.7"

ENTRYPOINT ["python"]
