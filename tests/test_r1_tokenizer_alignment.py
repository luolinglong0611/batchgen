"""Actual R1 tokenizer parity, including the worker's native batched path."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


class R1TokenizerAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        from batchgen.config.tokenizer_registry import load_tokenizer
        cls.service = load_tokenizer("deepseek-ai/DeepSeek-R1")
        assets = ROOT / "batchgen/models/deepseek/deepseekv3"
        cls.hf = AutoTokenizer.from_pretrained(str(assets), local_files_only=True)
        spec = importlib.util.spec_from_file_location("r1_tokenizer_qualification", ROOT / "scripts/r1_reproduction/qualify_r1_full_model.py")
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)
        cls.reference = cls.tool.reference_tokenizer(assets / "tokenizer.json")

    def worker(self, texts):
        return self.service(texts, return_tensors=None, truncation=False,
                            padding=False, return_attention_mask=False)["input_ids"]

    def test_plain_reference_single_and_worker_batch_match_released_hf(self):
        texts = ["Hello world", "", "你好，世界！", self.tool.render("What is 7 plus 8?")]
        expected = [self.hf.encode(text, add_special_tokens=True) for text in texts]
        self.assertTrue(all(ids[0] == 0 for ids in expected))
        self.assertEqual(self.worker(texts), expected)
        self.assertEqual([self.service.encode(text) for text in texts], expected)
        self.assertEqual([self.reference.encode(text).ids for text in texts], expected)

    def test_chat_and_existing_bos_have_one_automatic_boundary(self):
        messages = [{"role": "user", "content": "Hello world"}]
        for generation_prompt in (False, True):
            expected = self.hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt)
            text = self.service.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
            self.assertEqual(self.worker([text])[0], expected)
            self.assertEqual(self.service.encode(text), expected)
            self.assertEqual(expected.count(0), 1)
        bos = self.service.bos_token
        plain = self.hf.encode("Hello world", add_special_tokens=False)
        for count in (1, 2):
            text = bos * count + "Hello world"
            self.assertEqual(self.worker([text])[0], [0] * count + plain)
            self.assertEqual(self.service.encode(text, add_special_tokens=False), [0] * count + plain)

    def test_length_boundaries_include_bos_and_exact_prompt_roundtrips(self):
        for length in (63, 64, 65, 129, 2048, 8192):
            text, ids = self.tool.exact_length_prompt(self.reference, length)
            prompt = self.tool.exact_text(self.reference, ids)
            self.assertEqual(len(ids), length)
            self.assertEqual(self.hf.encode(prompt), ids)
            self.assertEqual(self.worker([prompt])[0], ids)


if __name__ == "__main__":
    unittest.main()
