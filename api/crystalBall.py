from transformers import pipeline
import re

class CrystalBall:

    def __init__(self):
        self.document = None
        self.nlp = pipeline("question-answering")

    def remove_markdown_syntax(self, text):
        # Remove markdown syntax and emojis
        text = re.sub(r'\!?\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'(:[^:\s]*:)', '', text)
        return text

    def read_document(self, file_path, doc_type='txt'):
        with open(file_path, 'r') as file:
            text = file.read()

        if doc_type.lower() == 'md':
            self.document = self.remove_markdown_syntax(text)
        else:
            self.document = text

    def answer_question(self, question):
        result = self.nlp(question=question, context=self.document, topk=3)
        return result

# Example usage
if __name__ == "__main__":
    file_path = '/Users/kjams/Desktop/Jiji/README.md'
    question = "What is the name of this API?"

    crystal_ball = CrystalBall()
    crystal_ball.read_document(file_path, doc_type='md')

    answers = crystal_ball.answer_question(question)

    print("Top 3 likely answers:")
    for answer in answers:
        print("Answer:", answer['answer'])
        print("Score:", answer['score'])
        print()
