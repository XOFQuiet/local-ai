# Local AI

A lightweight, private framework designed for running large language models and AI experiments directly on consumer hardware without cloud dependencies.

**Features**

* **100% Offline & Private:** Complete local execution ensuring your data never leaves your machine.
* **Hardware Optimized:** Tailored to efficiently utilize local CPU and GPU resources.
* **Open Architecture:** Easily plug in and swap out different open-weight model checkpoints.
* **Simple Interface:** Streamlined setup designed to get local inference up and running in minutes.

**Project Structure**

```text
local-ai/
├── src/          # Core inference and application logic
├── models/       # Model configuration and weight management
├── scripts/      # Automation and setup utilities
└── main.py       # Application entry point

```

**Getting Started**

1. Clone the repository:
```bash
git clone https://github.com/XOFQuiet/local-ai.git

```


2. Navigate to the project directory:
```bash
cd local-ai

```


3. Install the required dependencies:
```bash
pip install -r requirements.txt

```


4. Run the application:
```bash
python main.py

```



**Configuration**
Model parameters, context windows, and hardware acceleration options can be fine-tuned inside the configuration files or via environment variables before launching.

**License**
Distributed under the MIT License. See `LICENSE` for more information.
