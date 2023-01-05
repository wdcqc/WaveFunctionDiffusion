# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

from wfd.webui import start_demo
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script for WaveFunctionDiffusion.")
    parser.add_argument(
        "--colab",
        action="store_true", 
        help="Use it in Google Colab.",
    )
    parser.add_argument(
        "--link_to_colab",
        action="store_true", 
        help="Add a link to Google Colab.",
    )
    args = parser.parse_args()
    start_demo(args)
