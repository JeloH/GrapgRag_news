"""
Toy GraphRAG-style fake-news checker (BBC as external knowledge)

What it does:
1) Takes BBC articles (text + date) as your external knowledge.
2) Extracts simple entities + relations into a knowledge graph.
3) Takes a user claim, builds a tiny claim-graph.
4) Retrieves relevant BBC subgraph by entity overlap (graph-based retrieval).
5) Checks for timeline / relation mismatches and prints a verdict.

This is a *prototype* (simple heuristics). For real use you’d swap the
NER/relation extraction with spaCy/transformers and store in Neo4j.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

import networkx as nx


# ----------------------------
# 0) Sample BBC "external knowledge"
#    Replace this with your scraped BBC dataset.
# ----------------------------

@dataclass
class Article:
    id: str
    date: str  # "YYYY-MM-DD"
    title: str
    text: str


BBC_ARTICLES: List[Article] = [
    Article(
        id="bbc_001",
        date="2026-01-12",
        title="Germany sets coal phase-out target for 2030",
        text="Germany announced a ban on new coal plants, aiming to phase out coal by 2030."
    ),
    Article(
        id="bbc_002",
        date="2026-02-20",
        title="France discusses petrol car phase-out timeline",
        text="France announced plans to phase out petrol cars by 2035, officials said."
    ),
    Article(
        id="bbc_003",
        date="2025-12-05",
        title="Canada denies nationwide EV ban rumors",
        text="Canada denied claims of banning electric vehicles. The government said no such ban exists."
    ),
]


# ----------------------------
# 1) Simple entity extraction (toy NER)
#    - Countries/Places: capitalized words from a whitelist
#    - Years: 4-digit years
#    - Topics: keyword matches
# ----------------------------

COUNTRY_ORG_WHITELIST = {
    "Germany", "France", "Canada", "BBC", "government", "officials"
}

TOPIC_KEYWORDS = {
    "coal plants": "CoalPlants",
    "coal": "Coal",
    "petrol cars": "PetrolCars",
    "electric vehicles": "ElectricVehicles",
    "EV": "ElectricVehicles",
    "ban": "Ban",
    "phase out": "PhaseOut",
    "denied": "Denied",
    "announced": "Announced",
}

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b", re.IGNORECASE)

def extract_years(text: str) -> List[str]:
    return YEAR_RE.findall(text)  # returns tuples because of group

def extract_years_fixed(text: str) -> List[str]:
    # Use finditer so we get the full match
    return [m.group(0) for m in YEAR_RE.finditer(text)]

def find_topics(text: str) -> Set[str]:
    low = text.lower()
    found = set()
    for k, v in TOPIC_KEYWORDS.items():
        if k.lower() in low:
            found.add(v)
    return found

def find_countries_orgs(text: str) -> Set[str]:
    # Simple: match whitelist tokens in a case-sensitive way for names,
    # and also allow "government" / "officials" case-insensitive.
    found = set()
    for w in COUNTRY_ORG_WHITELIST:
        if w.lower() in {"government", "officials"}:
            if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
                found.add(w.title())
        else:
            if re.search(rf"\b{re.escape(w)}\b", text):
                found.add(w)
    return found


# ----------------------------
# 2) Relation extraction (toy)
#    We’ll extract edges like:
#      (Country) -[ANNOUNCED_BAN]-> (Topic)
#      (Topic)   -[YEAR]-> (2030)
# ----------------------------

def extract_relations(article: Article) -> List[Tuple[str, str, str, Dict]]:
    """
    Returns edges: (src, relation, dst, props)
    """
    edges = []
    text = f"{article.title}. {article.text}"

    years = extract_years_fixed(text)
    countries = find_countries_orgs(text)
    topics = find_topics(text)

    # Detect action type
    low = text.lower()
    action = None
    if "deny" in low or "denied" in low:
        action = "DENIED"
    elif "announce" in low or "announced" in low:
        action = "ANNOUNCED"
    if "ban" in low and action == "ANNOUNCED":
        action = "ANNOUNCED_BAN"
    elif "ban" in low and action is None:
        action = "BAN"

    # Connect country -> topic via action
    if action and countries and topics:
        for c in countries:
            for t in topics:
                edges.append((c, action, t, {"source": article.id, "date": article.date}))

    # Attach years to relevant topics if present
    if years and topics:
        for t in topics:
            for y in years:
                edges.append((t, "YEAR", y, {"source": article.id, "date": article.date}))

    return edges


# ----------------------------
# 3) Build BBC knowledge graph
# ----------------------------

def build_knowledge_graph(articles: List[Article]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for a in articles:
        G.add_node(a.id, type="ARTICLE", date=a.date, title=a.title)

        edges = extract_relations(a)
        for src, rel, dst, props in edges:
            # Ensure nodes exist
            if src not in G:
                G.add_node(src, type="ENTITY")
            if dst not in G:
                node_type = "YEAR" if dst.isdigit() and len(dst) == 4 else "ENTITY"
                G.add_node(dst, type=node_type)

            # Add edge
            G.add_edge(src, dst, relation=rel, **props)

            # Link edge to article node (provenance)
            # (article) -[MENTIONS]-> (src/dst)
            G.add_edge(a.id, src, relation="MENTIONS", date=a.date)
            G.add_edge(a.id, dst, relation="MENTIONS", date=a.date)

    return G


# ----------------------------
# 4) Build claim graph (toy)
# ----------------------------

def build_claim_graph(claim: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    text = claim

    years = extract_years_fixed(text)
    countries = find_countries_orgs(text)
    topics = find_topics(text)

    low = text.lower()
    # infer action from claim
    action = None
    if "denied" in low or "deny" in low:
        action = "DENIED"
    elif "announced" in low or "announce" in low:
        action = "ANNOUNCED"
    if "ban" in low:
        action = "BAN" if action is None else f"{action}_BAN"

    # Add edges similar to BBC graph
    if action and countries and topics:
        for c in countries:
            for t in topics:
                G.add_node(c, type="ENTITY")
                G.add_node(t, type="ENTITY")
                G.add_edge(c, t, relation=action, source="CLAIM")

    if years and topics:
        for t in topics:
            for y in years:
                G.add_node(t, type="ENTITY")
                G.add_node(y, type="YEAR")
                G.add_edge(t, y, relation="YEAR", source="CLAIM")

    return G


# ----------------------------
# 5) Graph-based retrieval: pull relevant BBC subgraph by entity overlap
# ----------------------------

def retrieve_relevant_subgraph(bbc_graph: nx.MultiDiGraph, claim_graph: nx.MultiDiGraph, hops: int = 2) -> nx.MultiDiGraph:
    claim_entities = {n for n, data in claim_graph.nodes(data=True) if data.get("type") in {"ENTITY", "YEAR"}}
    seed_nodes = [n for n in claim_entities if n in bbc_graph]

    # Expand neighborhood
    nodes = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            for nbr in bbc_graph.successors(n):
                next_frontier.add(nbr)
            for nbr in bbc_graph.predecessors(n):
                next_frontier.add(nbr)
        next_frontier -= nodes
        nodes |= next_frontier
        frontier = next_frontier

    return bbc_graph.subgraph(nodes).copy()


# ----------------------------
# 6) Consistency checks: timeline + relation checks
# ----------------------------

def edge_set(G: nx.MultiDiGraph) -> Set[Tuple[str, str, str]]:
    """(src, rel, dst) ignoring multi-edge keys/props"""
    s = set()
    for u, v, data in G.edges(data=True):
        s.add((u, data.get("relation"), v))
    return s

def check_claim_against_bbc(claim: str, bbc_graph: nx.MultiDiGraph) -> Dict:
    cg = build_claim_graph(claim)
    sub = retrieve_relevant_subgraph(bbc_graph, cg, hops=2)

    claim_edges = edge_set(cg)
    bbc_edges = edge_set(sub)

    # Quick signals
    missing = claim_edges - bbc_edges

    # Timeline mismatch check for (Topic) -YEAR-> (YYYY)
    claim_year_edges = {(u, v) for (u, rel, v) in claim_edges if rel == "YEAR" and str(v).isdigit()}
    bbc_year_edges = {(u, v) for (u, rel, v) in bbc_edges if rel == "YEAR" and str(v).isdigit()}

    timeline_mismatch = []
    for topic, y in claim_year_edges:
        # find any BBC year for that topic
        bbc_years = sorted({int(by) for (bt, by) in bbc_year_edges if bt == topic})
        if bbc_years and int(y) not in bbc_years:
            timeline_mismatch.append((topic, int(y), bbc_years))

    # Denial check: if BBC has DENIED for same (country/topic), claim says BAN
    denial_conflict = []
    for (u, rel, v) in claim_edges:
        if rel in {"BAN", "ANNOUNCED_BAN"}:
            # if BBC says DENIED between same u->v
            if (u, "DENIED", v) in bbc_edges:
                denial_conflict.append((u, v))

    # Decide verdict (simple rules)
    if denial_conflict:
        verdict = "LIKELY FALSE"
        reason = f"BBC subgraph contains DENIED for: {denial_conflict}"
    elif timeline_mismatch:
        verdict = "LIKELY FALSE / MISLEADING"
        reason = f"Timeline mismatch: {timeline_mismatch}"
    elif not missing and claim_edges:
        verdict = "SUPPORTED"
        reason = "Claim relations found in BBC subgraph."
    elif claim_edges:
        verdict = "UNVERIFIED"
        reason = f"Some claim relations not found in BBC subgraph: {list(missing)[:5]}"
    else:
        verdict = "UNVERIFIED"
        reason = "Could not extract structured relations from claim (try adding country + action + year)."

    # Collect evidence articles connected to subgraph nodes
    evidence_articles = []
    for n, data in sub.nodes(data=True):
        if data.get("type") == "ARTICLE":
            evidence_articles.append((n, data.get("date"), data.get("title")))
    evidence_articles.sort(key=lambda x: x[1], reverse=True)

    return {
        "verdict": verdict,
        "reason": reason,
        "claim_graph_edges": sorted(list(claim_edges)),
        "evidence_articles": evidence_articles[:5],
        "subgraph_nodes": sub.number_of_nodes(),
        "subgraph_edges": sub.number_of_edges(),
    }


# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    bbcG = build_knowledge_graph(BBC_ARTICLES)

    claims = [
        "Germany banned coal plants in 2025.",
        "France announced a phase out of petrol cars by 2035.",
        "Canada banned electric vehicles in 2026.",
    ]

    for c in claims:
        out = check_claim_against_bbc(c, bbcG)
        print("\nCLAIM:", c)
        print("VERDICT:", out["verdict"])
        print("REASON :", out["reason"])
        print("EVIDENCE (BBC):")
        for aid, d, t in out["evidence_articles"]:
            print(f"  - {d} | {aid} | {t}")
        print("CLAIM EDGES:", out["claim_graph_edges"])
        print(f"Subgraph size: nodes={out['subgraph_nodes']} edges={out['subgraph_edges']}")
