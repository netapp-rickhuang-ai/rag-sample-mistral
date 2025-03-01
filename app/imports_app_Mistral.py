from app.Mistral_transformers_pipeline import tokenizer, retriever, model, input_text
import unittest
from unittest.mock import patch  # For mocking expensive operations


class TestModel(unittest.TestCase):

    def test_tokenizer_not_none(self):
        self.assertIsNotNone(tokenizer)

    def test_retriever_not_none(self):
        self.assertIsNotNone(retriever)

    def test_model_not_none(self):
        self.assertIsNotNone(model)

    def test_input_text_not_none(self):
        self.assertIsNotNone(input_text)

    def test_input_text_is_string(self):
        self.assertIsInstance(input_text, str)

    def test_input_text_not_empty(self):
        self.assertTrue(len(input_text) > 0)

    def test_tokenizer_encode_decode(self):
        encoded = tokenizer(input_text, return_tensors="pt").input_ids
        decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded[0], input_text)

    def test_model_generate(self):
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, num_beams=2, num_return_sequences=1)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertIsNotNone(output)
        self.assertIsInstance(output, list)
        self.assertTrue(len(output) > 0)


if __name__ == '__main__':
    unittest.main()
