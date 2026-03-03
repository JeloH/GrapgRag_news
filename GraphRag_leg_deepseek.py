import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1) MANUAL DATA: put your legal sections/paragraphs here ---
DOCS = [
    {
        "id": "ActA_s12_1",
        "text": "Section 12(1): A person must ... (paste your statute text here)."
    },
    {
        "id": "Case_Smith_2019_p34_38",
        "text": "R v Smith (2019) paras 34–38: The court held that ... (paste here)."
    },
    # add more entries...
]

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # small HF DeepSeek model
EMB_ID   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K    = 5

def build_index(docs):
    embed = SentenceTransformer(EMB_ID)
    texts = [d["text"] for d in docs]
    metas = [d["id"] for d in docs]

    vecs = embed.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, texts, metas, embed

def generate_answer(question, sources):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).eval()

    prompt = (
        "You are a careful legal Q&A assistant.\n"
        "Use ONLY the Sources. If not in Sources, say you cannot find it.\n"
        "Cite like [1], [2] after the sentence.\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{sources}\n\n"
        "Answer:\n"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=350, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("Answer:\n", 1)[-1].strip()

def ask(question, index, texts, metas, embed):
    qv = embed.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)
    scores, ids = index.search(qv, TOP_K)

    blocks = []
    for rank, idx in enumerate(ids[0], 1):
        blocks.append(f"[{rank}] {metas[idx]}\n{texts[idx]}")
    sources = "\n\n---\n\n".join(blocks)

    print("\n=== SOURCES USED ===")
    for rank, (idx, s) in enumerate(zip(ids[0], scores[0]), 1):
        print(f"[{rank}] score={s:.3f} {metas[idx]}")

    print("\n=== ANSWER ===\n")
    print(generate_answer(question, sources))

if __name__ == "__main__":
    index, texts, metas, embed = build_index(DOCS)
    while True:
        q = input("\nQuestion (enter to quit): ").strip()
        if not q:
            break
        ask(q, index, texts, metas, embed)
