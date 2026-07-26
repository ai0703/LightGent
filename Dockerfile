FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# LightGent research service. Config comes from fly secrets (LLM_BASE_URL points
# at the balance broker, LLM_API_KEY is the broker bearer token).
# Bind IPv6 dual-stack (::) so Fly private networking (lightgent-app.internal,
# which is IPv6-only) can reach it. On Linux :: also accepts IPv4, so fly-proxy
# (public) still works.
CMD ["python", "-m", "uvicorn", "lightgent_service:app", "--host", "::", "--port", "8100", "--workers", "1"]
