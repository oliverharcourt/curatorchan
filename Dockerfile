FROM python:3.11.7-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry's configuration:
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.8.4 \
    PATH="/usr/local:$PATH"

# Install poetry
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME POETRY_VERSION=$POETRY_VERSION python3 -

# Copy the dep files
WORKDIR /app
COPY poetry.lock pyproject.toml ./

# Copy subomdule
COPY curatorchan/anime-recommender curatorchan/anime-recommender

# Install deps and copy source code
RUN poetry install --no-cache --no-root --only main --no-interaction --no-ansi && \
    rm -rf $POETRY_CACHE_DIR

COPY . .

# Start bot
CMD ["poetry", "run", "python", "curatorchan/bot/main.py"]
