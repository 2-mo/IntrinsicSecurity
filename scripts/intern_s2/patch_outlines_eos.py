"""Fix vLLM 0.22.1 rejecting EOS after Outlines accepts a complete JSON value."""
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


def main():
    if version('vllm') != '0.22.1':
        raise RuntimeError('Review the Outlines EOS compatibility patch for this vLLM version')
    path = Path(find_spec('vllm').origin).parent / 'v1/structured_output/backend_outlines.py'
    original = path.read_text()
    marker = '# Intern-S2: accept model stop tokens only after the grammar is complete.'
    if marker in original:
        print('Outlines EOS compatibility patch already applied')
        return
    changes = [
        ('        return OutlinesGrammar(\n            vocab_size=self.vocab_size,\n',
         '        generation = self.vllm_config.model_config.try_get_generation_config()\n'
         '        stop_ids = generation.get("eos_token_id", [])\n'
         '        if isinstance(stop_ids, int):\n'
         '            stop_ids = [stop_ids]\n'
         '        stop_ids = set(stop_ids or []) | {self.tokenizer.eos_token_id}\n'
         '        return OutlinesGrammar(\n            vocab_size=self.vocab_size,\n'
         '            stop_token_ids=tuple(t for t in stop_ids if isinstance(t, int)),\n'),
        ('    guide: oc.Guide = field(hash=False)\n',
         '    guide: oc.Guide = field(hash=False)\n'
         '    stop_token_ids: tuple[int, ...] = ()\n'),
        ('        if self.guide.accepts_tokens(tokens):\n',
         f'        {marker}\n'
         '        if self.guide.is_finished() and all(t in self.stop_token_ids for t in tokens):\n'
         '            self.num_processed_tokens += len(tokens)\n'
         '            return True\n'
         '        if self.guide.accepts_tokens(tokens):\n'),
    ]
    patched = original
    for before, after in changes:
        if patched.count(before) != 1:
            raise RuntimeError('Unexpected Outlines source; refusing to patch')
        patched = patched.replace(before, after, 1)
    compile(patched, str(path), 'exec')
    backup = Path(__file__).resolve().parents[2] / 'outputs/intern_s2/runtime/backend_outlines.py.original'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(original)
    path.write_text(patched)
    print('Applied Outlines EOS compatibility patch')


if __name__ == '__main__':
    main()
