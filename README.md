# 🧙‍♀️🔮 CrystalBall 🌙✨

![img1](api/imgs/jij.png)

Welcome to the magical CrystalBall! This Python class allows you to harness the power of divination and answer questions using the HuggingFaces NLP (Natural Language Processing) LLM (Large Language Models)

## Overview 🌟

CrystalBall is a versatile tool that utilizes the incredible capabilities of Hugging Face's Transformers library for question-answering tasks. With CrystalBall, you can unlock the secrets hidden within your documents and obtain answers to your burning questions.

## Installation 🧪🔬

To activate the enchanting powers of CrystalBall, follow these steps:

1. 🧹 Create and activate a virtual environment:
   ```shell
   python3 -m venv myenv
   source myenv/bin/activate
   ```


2. ✨ Install the required dependencies

    ```shell
    pip3 install -r requirements.txt
    ```
## Usage 🌟

To explore the mystical powers of CrystalBall, embark on a journey of question-answering with the following steps:

1. 🔮 Prepare your document:
   - Ensure your document is in the desired format (e.g., text or Markdown).
   - Place your document in a suitable location.

2. 🔍 Seek answers from CrystalBall:
   - Create an instance of the CrystalBall class in your Python script.
   - Use the `read_document()` method to provide CrystalBall with the path to your document.
   - Invoke the `answer_question()` method with your question of interest.
   - Marvel at the wisdom and insights revealed by CrystalBall.

Here's an example to get you started:

```python
from transformers import pipeline
import re

class CrystalBall:
    # ...

# Example usage
if __name__ == "__main__":
    file_path = '/path/to/document.md'
    question = "What is the main topic of the document?"

    crystal_ball = CrystalBall()
    crystal_ball.read_document(file_path, doc_type='md')

    answers = crystal_ball.answer_question(question)

    print("Top 3 likely answers:")
    for answer in answers:
        print("Answer:", answer['answer'])
        print("Score:", answer['score'])
        print()
```

May the magic of CrystalBall guide you on your quest for knowledge and insights! ✨🔮🌙
