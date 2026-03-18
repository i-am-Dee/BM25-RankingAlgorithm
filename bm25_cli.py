# bm25_cli.py
from bm25_simple import BM25

def read_corpus(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(line)
    return docs

def main():
    path = "corpus.txt"  # one document per line
    raw_corpus = read_corpus(path)
    tokenized = [BM25.simple_tokenize(doc) for doc in raw_corpus]
    bm25 = BM25(tokenized, k1=1.5, b=0.75)

    print("Type a query and press Enter. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuery> ").strip()
        except EOFError:
            break
        if not q or q.lower() == "exit":
            break
        ranked = bm25.rank(q, topk=5)
        for idx, score in ranked:
            print(f"  Score={score: .4f} | Doc[{idx}]: {raw_corpus[idx]}")

if __name__ == "__main__":
    main()