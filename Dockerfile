# Pinned environment for exact reproduction: docker build -t critcloud . && docker run --rm -v "$PWD":/w critcloud
FROM python:3.13.16-slim
WORKDIR /w
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["bash", "run_all.sh"]
