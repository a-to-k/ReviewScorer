import json
import os
import sys
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Globale Konfiguration (In-Memory ChromaDB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3", temperature=0.0)

_in_memory_db = None


def get_db():
  global _in_memory_db
  if _in_memory_db is None:
    _in_memory_db = Chroma(embedding_function=embeddings)
  return _in_memory_db


# 1. 100 Reviews aus der JSON-Datei in den RAM laden
def load_training_data_from_json():
  json_path = "reviews_100.json"
  if not os.path.exists(json_path):
    print(f"❌ Die Datei '{json_path}' wurde im Ordner nicht gefunden!")
    return

  print(f"Lade Trainingsdaten aus '{json_path}' in den Arbeitsspeicher...")
  with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  documents = []
  for item in data:
    doc = Document(
        page_content=item["text"],
        metadata={
            "title": item["title"],
            "score": item["score"],
            "id": item["id"],
        },
    )
    documents.append(doc)

  db = get_db()
  db.add_documents(documents)
  print(
      f"✅ Erfolgreich {len(documents)} Reviews in die In-Memory-ChromaDB"
      " geladen!"
  )


# 2. Review testen und bewerten (mit Human-in-the-Loop)
def evaluate_new_review():
  print("\n--- NEUE REVIEW BEWERTEN ---")
  print(
      "Gib den Text der Review ein (oder tippe 'default' für einen"
      " Test-Text):"
  )
  user_input = input("> ").strip()

  if user_input.lower() == "default" or not user_input:
    # Ein Test-Text, der perfekt zum Action-Cluster passen sollte
    new_text = (
        "Ein absolut rasanter Actionfilm mit genialen Effekten und starkem"
        " Soundtrack. Die Story ist zwar dünn, aber das Tempo stimmt."
    )
    print(f"\n[Verwende Test-Text]: {new_text}\n")
  else:
    new_text = user_input

  db = get_db()
  retriever = db.as_retriever(search_kwargs={"k": 3})
  relevant_docs = retriever.invoke(new_text)

  if not relevant_docs:
    (
        print(
            "⚠️ Keine Referenzen in der DB. Bitte zuerst Option 1 (Daten"
            " laden) ausführen!"
        )
    )
    return

  context_str = "\n\n".join(
      [
          (
              f"Referenz-Film: {doc.metadata.get('title')} (Score:"
              f" {doc.metadata.get('score')}):\n{doc.page_content}"
          )
          for doc in relevant_docs
      ]
  )

  prompt = f"""Du bist ein Filmkritiker-Assistent. Vergleiche die neue Review mit den historischen Referenzen.
Gib einen Score (0 bis 10) basierend auf den Ähnlichkeiten aus und bewerte deine Konfidenz als "HOCH" oder "NIEDRIG".

ÄHNLICHE REFERENZEN:
{context_str}

NEUE REVIEW:
{new_text}

Antworte im exakten Format:
SCORE: X.X
KONFIDENZ: [HOCH oder NIEDRIG]
BEGRÜNDUNG: [Kurzer Text]
"""

  print("Analysiere Review mit lokalem LLM...")
  response = llm.invoke(prompt)
  print("\n--- TEXT DER ÄHNLICHSTEN REFERENZEN ---")
  print(context_str)
  print("\n--- LLM ERGEBNIS ---")
  print(response)

  # Human-in-the-Loop
  if "KONFIDENZ: NIEDRIG" in response.upper():
    print(
        "\n⚠️ [HUMAN-IN-THE-LOOP] Das LLM ist sich unsicher oder findet keine"
        " eindeutige Basis."
    )
    choice = (
        input("Möchtest du den Score manuell korrigieren und lernen? (j/n): ")
        .strip()
        .lower()
    )
    if choice == "j":
      correct_score = float(input("Bitte echten Score eingeben: ").strip())
      new_doc = Document(
          page_content=new_text,
          metadata={
              "title": "Manuell korrigierte Review",
              "score": correct_score,
          },
      )
      db.add_documents([new_doc])
      print(
          "✅ Korrigierter Score wurde für diese Sitzung in den RAM gelernt!"
      )


# 3. Hauptmenü
def main():
  # Automatisch beim Start die 100 Reviews in den RAM laden
  load_training_data_from_json()

  while True:
    print("\n==============================")
    print("   FILM REVIEW EVALUATOR CLI")
    print("==============================")
    print("1. Neue Review testen / bewerten")
    print("2. Beenden")

    choice = input("Wähle eine Option (1-2): ").strip()
    if choice == "1":
      evaluate_new_review()
    elif choice == "2":
      print("Programm beendet. Arbeitsspeicher wurde bereinigt.")
      sys.exit(0)


if __name__ == "__main__":
  main()