FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY frontend/ ./frontend/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]