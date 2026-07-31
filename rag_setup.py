import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("disaster_docs")
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def ingest_doc(doc_id, text_chunks, disaster_type):
    embeddings = embedder.encode(text_chunks).tolist()
    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        metadatas=[{"disaster_type": disaster_type} for _ in text_chunks],
        ids=[f"{doc_id}_{i}" for i in range(len(text_chunks))]
    )


def query_rag(query, disaster_type, n_results=3):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
        where={"disaster_type": disaster_type}
    )
    docs = results.get('documents', [[]])
    return docs[0] if docs else []


def seed_default_docs():
    """Loads starter guidance so the RAG store isn't empty on first run."""
    earthquake_chunks = [
        "During an earthquake, drop, cover, and hold on immediately. Get under sturdy furniture and hold on until the shaking stops.",
        "Stay away from windows, mirrors, and heavy furniture or appliances that could fall or topple over.",
        "If outdoors during an earthquake, move to an open area away from buildings, trees, streetlights, and power lines.",
        "If indoors, do not run outside during shaking. Most injuries occur when people try to move to a different location.",
        "After the shaking stops, check yourself and others for injuries. Expect aftershocks and be ready to drop, cover, and hold on again.",
        "If trapped under debris, do not shout unless necessary. Tap on a pipe or wall so rescuers can locate you, and cover your mouth to avoid dust.",
        "Turn off gas if you smell leaks or hear hissing after an earthquake, and avoid using open flames.",
        "Before an earthquake, secure heavy furniture to walls, know where your gas/water/electricity shutoffs are, and keep an emergency kit ready.",
    ]
    cyclone_chunks = [
        "As a cyclone approaches, secure loose objects outdoors, board up windows, and move vehicles to higher, sheltered ground.",
        "Stock at least three days of drinking water, non-perishable food, medicines, flashlights, and a battery or hand-crank radio before landfall.",
        "Move to a designated cyclone shelter or the strongest interior room of a sturdy building, away from windows, before winds intensify.",
        "Do not go outside during the calm 'eye' of the cyclone. Winds will resume suddenly and violently from the opposite direction.",
        "Avoid coastal areas, riverbanks, and low-lying regions due to storm surge and flash flooding risk during a cyclone.",
        "After the cyclone passes, avoid downed power lines, standing floodwater, and damaged structures until authorities confirm the area is safe.",
        "Keep mobile phones charged and conserve battery; use SMS instead of calls where possible since networks may be congested.",
        "Follow official evacuation orders promptly; do not wait until conditions worsen to leave vulnerable areas.",
    ]
    ingest_doc("earthquake_default", earthquake_chunks, "earthquake")
    ingest_doc("cyclone_default", cyclone_chunks, "cyclone")
    print("Seeded default earthquake + cyclone guidance into RAG store.")


if __name__ == "__main__":
    seed_default_docs()
    print(query_rag("what should I do during shaking", "earthquake"))
    print(query_rag("what should I do before the storm hits", "cyclone"))
