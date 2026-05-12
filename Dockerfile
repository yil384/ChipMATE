FROM python:3.11-slim

# iverilog + vvp are required by chipmate.cross_verify to compile and run the
# Verilog DUT against random stimuli. The slim Python base image is missing
# them, so we install Icarus Verilog from the Debian repositories.
RUN apt-get update \
 && apt-get install -y --no-install-recommends iverilog ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY pyproject.toml requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package + examples.
COPY chipmate /app/chipmate
COPY examples /app/examples
COPY README.md LICENSE /app/

RUN pip install --no-cache-dir .

ENTRYPOINT ["chipmate"]
CMD ["--help"]
