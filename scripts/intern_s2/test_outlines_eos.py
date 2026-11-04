"""Regression checks with the real Intern-S2 tokenizer; run in .venv-intern-s2."""
import json
import unittest

import outlines_core as oc
import outlines_core.json_schema as json_schema
from transformers import AutoTokenizer
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.v1.structured_output.backend_outlines import OutlinesGrammar
from vllm.v1.structured_output.utils import get_outlines_vocabulary


class OutlinesEOSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_cached_tokenizer(AutoTokenizer.from_pretrained(
            '/data3/Intern-S2-Preview', trust_remote_code=True))
        vocabulary = get_outlines_vocabulary(cls.tokenizer)
        schema = {'type': 'object', 'properties': {
            'reasoning': {'type': 'string'},
            'final_answer': {'type': 'string', 'enum': ['True', 'False', 'Unknown']},
        }, 'required': ['reasoning', 'final_answer'], 'additionalProperties': False}
        cls.index = oc.Index(json_schema.build_regex_from_schema(json.dumps(schema)), vocabulary.inner)

    def grammar(self):
        return OutlinesGrammar(vocab_size=251392, guide=oc.Guide(self.index, max_rollback=0),
                               stop_token_ids=(248046, 248044))

    def test_model_stop_tokens_are_accepted_after_complete_json(self):
        tokens = self.tokenizer.encode('{"reasoning":"test","final_answer":"True"}',
                                       add_special_tokens=False)
        for eos in (248046, 248044):
            with self.subTest(eos=eos):
                grammar = self.grammar()
                self.assertTrue(grammar.accept_tokens('test', tokens))
                self.assertTrue(grammar.accept_tokens('test', [eos]))

    def test_stop_tokens_are_rejected_before_json_is_complete(self):
        self.assertFalse(self.grammar().accept_tokens('test', [248046]))

    def test_schema_still_rejects_invalid_answer_labels(self):
        tokens = self.tokenizer.encode('{"reasoning":"test","final_answer":"E"}',
                                       add_special_tokens=False)
        self.assertFalse(self.grammar().accept_tokens('test', tokens))


if __name__ == '__main__':
    unittest.main()
