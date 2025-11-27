# Python AI Project

This repository contains miscellaneous AI code and projects.

## Description

This project is a collection of Python scripts and Jupyter notebooks related to Artificial Intelligence, Machine Learning, and Large Language Models (LLMs). It includes exercises and examples from the "LLM Engineering - Master AI and LLMs" Udemy course.

## Notebooks

*   **`week1-Day1-Text-Subject-Summary-UrduTranslaction.ipynb`**: Week 1, Day 1 exercise from the course. This notebook contains code for text summarization and translation into Urdu.
*   **`week1 EXERCISE - EngineeringTutor.ipynb`**: Week 1 exercise from the course. This notebook implements an "Engineering Tutor" using LLMs.

## Setup

To set up the project environment, follow these steps:

1.  **Install uv**: If you don't have `uv` installed, follow the instructions at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

2.  **Check uv version**:
    ```bash
    uv --version
    ```

3.  **Update uv**:
    ```bash
    uv self update
    ```

4.  **Pin Python version (optional)**: If you need to use a specific Python version, you can pin it. For example:
    ```bash
    uv python pin 3.12
    ```

5.  **Sync dependencies**: Install the project dependencies using `uv`.
    ```bash
    uv sync
    ```