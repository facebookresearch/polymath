# ChangePr

## Refactor: Modular Engine Strategy, Prompt Revision and Error Handling

---

### Engine Strategy System

- **Created a new engine strategy: `PrologEngineStrategy`**
  - Implements the existing `EngineStrategy` interface.
  - Enables solving logic puzzles using a Prolog backend.

- **Introduced a `StrategyFactory`**
  - Allows injecting different `EngineStrategy` implementations dynamically.
  - Replaces the hardcoded use of `CBMCEngineStrategy` in the main logic.

---

### Solution Comparator System

- **Added an abstract class `SolutionComparator`**
  - Defines a generic interface for comparing expected vs actual solutions.

- **Implemented `ZebraSolutionComparator`**
  - Specialized comparator for Zebra-like logic puzzles.
  - Compares solutions and provides:
    - Success or failure
    - Number of mismatches
    - Detailed error messages when differences are found

---

### Prompt Revision System

- **Created an abstract class `PromptReviser`**
  - Defines the interface for revising or adapting prompts in case of failure.
  - Designed to allow multiple prompt revision strategies.

- **Implemented `AdaptivePromptReviser`**
  - Extends `PromptReviser`.
  - Receives the original prompt, generated code, and formatted error.
  - Iteratively generates improved prompts based on model feedback.
  - Temperature is adjusted during iterations for better exploration.
  - Designed to be passed directly to the `LogicAgent` pipeline.

- **Implemented `NoOpPromptReviser`**
  - A minimal implementation of `PromptReviser` that performs no revisions.
  - Used as a default when prompt revision is not required.
  - Prevents the need for None checks or conditional logic throughout the agent.
  - Fully compatible with the agent pipeline thanks to polymorphism (Null Object Pattern).

---

### Error Handling System

- **Added an `ErrorHandler` class**
  - Takes raw solver errors (e.g., from CBMC) and formats them into natural language.
  - Output is designed to be interpretable by the reviser LLM.
  - Helps guide the prompt adaptation process more effectively.

---

### Multi-Backend ChatCompletion Interface

- **Implemented `MistralChatCompletion`**
  - Works with local models using the `mistral_inference` library.
  - Requires no API key or internet access.
  - Designed for performant inference on compatible local hardware.

- **Implemented `GroqChatCompletion`**
  - Interfaces with the Groq API (e.g., LLaMA 3, Mistral).
  - Requires an API key.
  - Allows usage of powerful hosted models with low-latency inference.

- **Implemented `LlamaChatCompletion`**
  - Works locally with Hugging Face Transformers.
  - Uses 4-bit quantization (`BitsAndBytesConfig`) with `AutoModelForCausalLM`.
  - Loads model using `pipeline("text-generation")`.
  - Compatible with GPU acceleration and memory-efficient quantized inference.
