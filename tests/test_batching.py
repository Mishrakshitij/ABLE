import unittest
from types import SimpleNamespace

from able.batching import batches, encode_prompt, encode_response, shuffled


class Tokenizer:
    eos_token_id = 99
    bos_token_id = 0
    pad_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]


class BatchingTests(unittest.TestCase):
    def test_response_only_mask_and_eos(self):
        data = encode_response(Tokenizer(), "1 2 3", "4 5", 10)
        self.assertEqual(data["input_ids"], [1, 2, 3, 4, 5, 99])
        self.assertEqual(data["labels"], [-100, -100, -100, 4, 5, 99])

    def test_long_prompt_keeps_prefix_and_recent_turn(self):
        data = encode_prompt(Tokenizer(), "1 2 3 4 5 6 7 8 9", 6)
        self.assertEqual(data, [1, 2, 6, 7, 8, 9])

    def test_long_response_preserves_context_and_eos(self):
        data = encode_response(Tokenizer(), "1 2 3", "4 5 6 7 8 9", 4)
        self.assertEqual(data["labels"], [-100, 4, 5, 99])
        self.assertEqual(encode_prompt(Tokenizer(), "", 4), [0])

    def test_shuffle_coverage_and_reproducibility(self):
        a = list(shuffled(range(101), 10, 4))
        self.assertEqual(sorted(a), list(range(101)))
        self.assertEqual(a, list(shuffled(range(101), 10, 4)))
        self.assertNotEqual(a, list(shuffled(range(101), 11, 4)))
        self.assertEqual(list(batches(range(5), 2)), [[0, 1], [2, 3], [4]])

    def test_padding_eos_is_not_supervised(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Install the train extra for tensor tests")
        from able.batching import collate_responses
        batch = collate_responses([SimpleNamespace(prompt="1 2", response="3"),
                                   SimpleNamespace(prompt="1", response="3")], Tokenizer(), 10)
        self.assertEqual(batch["input_ids"][1, -1].item(), 99)
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["labels"][1, -2].item(), 99)


if __name__ == "__main__":
    unittest.main()
