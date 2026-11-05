# What this file does: the command line: `python -m groundedqa ask <file> "<question>"`
# prints the answer and its receipts as JSON, with no server at all.
"""python -m groundedqa ask data/handbook.md "what happens to lost gear?" """

from __future__ import annotations

import json
import sys

from .ingest import RefusedDocument
from .store import Store


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "ask":
        print('usage: python -m groundedqa ask <file> "<question>"', file=sys.stderr)
        return 2
    path, question = argv[1], argv[2]
    try:
        with open(path, "rb") as fh:
            doc = Store().add(fh.read(), path)
    except (OSError, RefusedDocument) as exc:
        print(json.dumps({"refused": True, "why": str(exc)}))
        return 1
    store = Store()
    store.docs[doc.id] = doc
    print(json.dumps(store.ask(doc.id, question).to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
