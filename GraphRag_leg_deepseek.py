import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1) MANUAL DATA: put your legal sections/paragraphs here ---

DOCS = [
    {
        "id": "TheftAct1968_s1",
        "text": (
            "Theft Act 1968, Section 1: A person is guilty of theft if he dishonestly "
            "appropriates property belonging to another with the intention of "
            "permanently depriving the other of it."
        )
    },
    {
        "id": "TheftAct1968_s2",
        "text": (
            "Theft Act 1968, Section 2: A person's appropriation of property belonging "
            "to another is not dishonest if he believes he has a legal right to deprive "
            "the other of it, would have the other's consent, or cannot discover the owner "
            "by taking reasonable steps."
        )
    },
    {
        "id": "R_v_Ghosh_1982",
        "text": (
            "R v Ghosh [1982]: The Court of Appeal established a two-stage test for "
            "dishonesty: (1) whether according to ordinary standards the conduct was dishonest, "
            "and (2) whether the defendant realized that reasonable and honest people "
            "would regard the conduct as dishonest."
        )
    },
    {
        "id": "Ivey_v_Genting_2017",
        "text": (
            "Ivey v Genting Casinos [2017] UKSC 67: The Supreme Court clarified that "
            "the test for dishonesty is objective. The fact-finder must ascertain the "
            "defendant's actual state of knowledge or belief as to the facts and then "
            "determine whether the conduct was dishonest by the standards of ordinary "
            "decent people."
        )
    },
    {
        "id": "R_v_Smith_1974_Property",
        "text": (
            "R v Smith (1974): The Court of Appeal held that property may still belong "
            "to another under the Theft Act even if the defendant had some proprietary "
            "interest in it, provided another person retained control or rights over it."
        )
    },
    {
        "id": "CriminalDamageAct1971_s1",
        "text": (
            "Criminal Damage Act 1971, Section 1(1): A person who without lawful excuse "
            "destroys or damages property belonging to another, intending to destroy or "
            "damage such property or being reckless as to whether such property would "
            "be destroyed or damaged, shall be guilty of an offence."
        )
    },
    {
        "id": "R_v_Blaue_1975",
        "text": (
            "R v Blaue [1975]: The Court of Appeal held that a defendant must take the "
            "victim as found (the thin skull rule). A victim's refusal of medical treatment "
            "for religious reasons does not break the chain of causation."
        )
    }
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
