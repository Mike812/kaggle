ARG PYTHON_VERSION=3.12.4
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    --mount=type=bind,source=xgb_mental_health.pkl,target=xgb_mental_health.pkl \
    --mount=type=bind,source=train_val_columns.csv,target=train_val_columns.csv \
    python -m pip install -r requirements.txt

COPY kaggle.tar.gz .

RUN tar -xzvf kaggle.tar.gz

RUN cd kaggle

WORKDIR .

ENV PYTHONPATH=.

RUN cd fast_api_app

# Expose the port that the application listens on.
EXPOSE 8000

USER appuser

# Run the application.
CMD python3 -m uvicorn fast_api_app:fast_api_app --host=0.0.0.0 --port=8001
