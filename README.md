# MMI-reranking DialoGPT
Implmentation of MMI-Reranking Dialo-GPT.

## Downloads

you have to download DialoGPT 345M model, DialoGPT 345M model (reverse), and config.

## Examples

```
Example Context: How are you doing?
Output Utterance Top-1: I'm good. You?
Output Utterance Top-2: How are you?
Output Utterance Top-3: I'll be okay
Output Utterance Top-4: Doin good too?
Output Utterance Top-5: Well..not that great.

Example Context: Does money buy happiness?<|endoftext|>Depends how much money you spend on it .
Output Utterance Top-1: So, money buys happiness
Output Utterance Top-2: No. Money buys happiness.
Output Utterance Top-3: Money buys happiness and a sense of humor.
Output Utterance Top-4: It's not enough.
Output Utterance Top-5: And a few years of debt!
```
