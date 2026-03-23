# Architecture

## Sequence format

Each training sample is a flat token sequence:

```
<s> <text> h ə l ˈoʊ <generate> <audio> a0 a1 a2 ... </s>
```

| Token | Role |
|---|---|
| `<s>` | start of sequence |
| `<text>` | marks start of IPA input |
| `h ə l ˈoʊ` | IPA phonemes, one token per character |
| `<generate>` | boundary — loss is masked up to here (inclusive) |
| `<audio>` | marks start of audio output |
| `a0 a1 ...` | SNAC codec tokens (with delay pattern applied) |
| `</s>` | end of sequence |

## Loss masking

The model only learns to predict the audio region:

```
<s> <text> h ə l ˈoʊ <generate>  |  <audio> a0 a1 a2 ... </s>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^
        masked (-100)                      loss computed here
```

## Audio tokens (SNAC)

SNAC produces 3 hierarchical code levels (c0, c1, c2). They are interleaved depth-first with a delay pattern (c1 delayed 1 step, c2 delayed 2 steps) to make the prediction easier — each token depends only on tokens already generated at coarser levels.

7 tokens per timestep: `c0 c1 c2 c2 c1 c2 c2`
