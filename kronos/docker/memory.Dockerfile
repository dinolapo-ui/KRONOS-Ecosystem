FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY libs/kronos_common /opt/kronos_common
RUN pip install --no-cache-dir /opt/kronos_common "chromadb>=1.0,<2"
COPY memory/app /srv/app
RUN useradd --create-home kronos && mkdir /data && chown kronos /data
USER kronos
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
