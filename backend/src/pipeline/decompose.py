import re
from mlx_lm import generate

_PROP_PATTERN = re.compile(r"<s>(.*?)</s>", re.DOTALL)

def _parse_propositions(text: str) -> list[str]:
    propositions = []
    for group in _PROP_PATTERN.findall(text):
        for line in group.strip().splitlines():
            line = line.strip().removeprefix("- ")
            if line:
                propositions.append(line)
    return propositions


# ---------------------------------------------------------------------------
# Fact / opinion classification  (lighteternal/fact-or-opinion-xlmr-el)
# ---------------------------------------------------------------------------

async def classify(
    sentences: list[str],
    classifier,      # transformers text-classification pipeline on app.state
    batch_size: int = 16,
    stop_flag: list[bool] | None = None,
) -> list[str]:
    """Return only the sentences labelled as facts with manual batching."""
    if stop_flag is None:
        stop_flag = [False]

    facts: list[str] = []

    for i in range(0, len(sentences), batch_size):
        if stop_flag[0]:
            break

        batch = sentences[i : i + batch_size]
        results = classifier(batch)

        for sentence, result in zip(batch, results):
            if result["label"] == "LABEL_1":
                facts.append(sentence)

    return facts


# ---------------------------------------------------------------------------
# Atomic proposition extraction  (google/gemma-2b-aps-it)
# ---------------------------------------------------------------------------

async def extract_propositions(
    sentences: list[str],
    aps,             # mlx model and tokeniser stores in app.state
    max_new_tokens: int = 1024,
    stop_flag: list[bool] | None = None,
) -> dict[str, list[str]]:
    """Return a mapping of sentence → list of atomic propositions with manual batching."""
    if stop_flag is None:
        stop_flag = [False]

    results: dict[str, list[str]] = {}
    
    model, tokeniser = aps[0], aps[1]

    for sentence in sentences:
        if stop_flag[0]:
            break
        
        prompt = tokeniser.apply_chat_template(
            [{"role": "user", "content": sentence}],
            tokenize=False,
            add_generation_prompt=True,
        )

        output = generate(model, tokeniser, prompt=prompt, max_tokens=max_new_tokens)

        results[sentence] = _parse_propositions(output)

    return results
