#!/usr/bin/env python3
"""Compatibility entry point; prefer the installed `sentiment-lab` command."""

from sentiment_lab.cli import app

if __name__ == "__main__":
    app()
