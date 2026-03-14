# src/mlshield/main.py
"""MLShield entry point."""

import click
import uvicorn
from .utils.config import load_config
from .utils.logging import setup_logging


@click.group()
def cli():
    """MLShield - ML Infrastructure Security Monitor."""
    pass


@cli.command()
@click.option("--config", default=None, help="Path to config YAML")
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, type=int, help="API port")
@click.option("--log-level", default="INFO", help="Log level")
def serve(config, host, port, log_level):
    """Start the MLShield monitoring server."""
    load_config(config)
    logger = setup_logging(level=log_level)
    logger.info("Starting MLShield", version="0.1.0", host=host, port=port)

    uvicorn.run(
        "mlshield.api.app:app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        reload=False,
    )


@cli.command()
@click.option("--output-dir", default="benchmark/data", help="Output directory")
@click.option(
    "--n-normal", default=1500, type=int, help="Number of normal trajectories"
)
@click.option("--n-exfil", default=200, type=int, help="Number of exfil trajectories")
def generate(output_dir, n_normal, n_exfil):
    """Generate synthetic benchmark dataset."""
    from benchmark.generate_dataset import generate_full_dataset

    generate_full_dataset(n_normal=n_normal, n_exfil=n_exfil, output_dir=output_dir)


if __name__ == "__main__":
    cli()
