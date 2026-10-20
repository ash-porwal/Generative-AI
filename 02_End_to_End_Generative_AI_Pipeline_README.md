# 🏗️ End-to-End Generative AI Pipeline — Beginner to PRO

> **Goal:** Understand every step needed to take a GenAI idea from **raw data** to a **working app in production**, plus how to keep it healthy afterwards.
> Every new term is explained in simple language the first time it appears.

---

## 📚 Table of Contents

1. [What is a "Pipeline"?](#1-what-is-a-pipeline)
2. [The Big Picture (All Steps at a Glance)](#2-the-big-picture)
3. [Step 1 — Data Acquisition](#3-step-1--data-acquisition)
4. [Step 2 — Data Augmentation](#4-step-2--data-augmentation)
5. [Step 3 — Data Preparation (Cleaning + Preprocessing)](#5-step-3--data-preparation)
6. [Step 4 — Feature Engineering (Text → Numbers)](#6-step-4--feature-engineering)
7. [Step 5 — Modeling](#7-step-5--modeling)
8. [Step 6 — Evaluation](#8-step-6--evaluation)
9. [Step 7 — Deployment](#9-step-7--deployment)
10. [Step 8 — Monitoring & Model Updating](#10-step-8--monitoring--model-updating)
11. [Full Mini Example: Customer-Review Pipeline](#11-full-mini-example)
12. [PRO Level: The Modern LLM App Pipeline (RAG)](#12-pro-level-the-modern-llm-app-pipeline-rag)
13. [PRO Level: LLMOps](#13-pro-level-llmops)
14. [Common Mistakes](#14-common-mistakes)
15. [Interview / Revision Cheat Sheet](#15-interview--revision-cheat-sheet)

---

## 1. What is a "Pipeline"?

A **pipeline** is a **series of steps** where the output of one step becomes the input of the next — like an assembly line.

🧒 **Cooking analogy:**
```
Buy vegetables → Wash → Chop → Cook → Taste → Serve → Get feedback → Improve recipe
   (acquire)    (clean) (prep) (model) (evaluate) (deploy)  (monitor)    (retrain)
```

As a data engineer, you already know ETL pipelines (Extract → Transform → Load). A GenAI pipeline is the same idea, but the end product is a **model / AI app** instead of a table.

**End-to-end** = covering **everything from start to finish**, not just one part.

---

## 2. The Big Picture

```
┌────────────────────┐
│ 1. Data Acquisition│  Get the data
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 2. Data Augmentation│ Create more data if you have too little
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 3. Data Preparation │ Clean + preprocess the text
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 4. Feature Eng.     │ Convert text → numbers
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 5. Modeling         │ Choose / train / fine-tune a model
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 6. Evaluation       │ Is it good enough?
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 7. Deployment       │ Make it available to users
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ 8. Monitoring &     │ Watch it, fix it, retrain it
│    Model Updating   │──────┐
└────────────────────┘      │
          ↑                  │
          └──── feedback ────┘  (loop back to step 1)
```

> 💡 The pipeline is a **loop**, not a straight line. Real projects go back and forth many times.

---

## 3. Step 1 — Data Acquisition

**Data acquisition** = collecting the data your model will learn from or use.

### Three situations you can be in

#### ✅ Situation A — Data is already available
Files you already have:
- `.csv` → tables (e.g., customer reviews)
- `.txt` → plain text
- `.pdf` → documents (policies, manuals)
- `.docx`, `.xlsx`, `.json`, `.html`

```python
import pandas as pd
df = pd.read_csv("reviews.csv")
print(df.head())
```

#### 🌐 Situation B — Data exists elsewhere (you must fetch it)
| Source | How | Example |
|---|---|---|
| **Database** | SQL query | Pull support tickets from PostgreSQL / Redshift |
| **Internet** | Download public datasets | Kaggle, Hugging Face Datasets |
| **API** | Call a service | Twitter/X API, company REST APIs |
| **Web scraping** | Extract text from web pages | `requests` + `BeautifulSoup` |
| **Data lake** | Read from cloud storage | S3, ADLS |

🧠 **New term: Web scraping** — writing code that visits web pages and pulls out text automatically. (Always respect website terms and legal rules.)

```python
import requests
from bs4 import BeautifulSoup

html = requests.get("https://example.com").text
text = BeautifulSoup(html, "html.parser").get_text()
```

#### ❌ Situation C — No data at all
Options:
1. **Manually create/label data** — humans write examples (slow, costly, high quality).
2. **Generate synthetic data with an LLM** — ask an LLM to produce examples.

🧠 **New term: Synthetic data** — artificially generated data that looks like real data.

```
Prompt to an LLM:
"Generate 20 short customer complaints about late food delivery,
 each with a label: angry / neutral / polite."
```
⚠️ Always review synthetic data — it can be repetitive or unrealistic.

### 🧠 New term: Dataset
A **dataset** is the collection of data you use (e.g., 10,000 reviews).

### 🧠 New term: Corpus
**Corpus** = a large collection of text used for NLP. "Wikipedia corpus" = all Wikipedia text.

### 🧠 New term: NLP
**NLP (Natural Language Processing)** = the field of AI that deals with human language (text and speech).

---

## 4. Step 2 — Data Augmentation

**Data augmentation** = creating **new training examples from existing ones** by making small changes. Used when you have **too little data**.

🧒 **Analogy:** You have 1 photo of a cat. Flip it, rotate it, brighten it → now you have 4 "different" cat photos for training.

### Text augmentation techniques

| Technique | What it does | Example |
|---|---|---|
| **Synonym replacement** | Swap words with similar meaning | "I am **happy**" → "I am **glad**" |
| **Bigram flip** | Swap the order of two neighbouring words where meaning stays OK | "I am going **to the** supermarket" → "I am going **the to** supermarket" *(use carefully — can break grammar)* |
| **Back translation** | Translate to another language and back | English → Hindi → English: "The food was very good" → "The meal was really good" |
| **Adding noise** | Insert small typos/random characters (mimics real user typing) | "I love this phone" → "I lvoe this phone" |
| **Random insertion / deletion** | Add or remove a non-important word | "This is a great movie" → "This is great movie" |
| **LLM paraphrasing** | Ask an LLM to rewrite sentences | "Rewrite in 3 different ways" |

### 🧠 New term: Bigram
A **bigram** = a pair of **2 consecutive words**. In "I love pizza": bigrams are "I love", "love pizza". (Unigram = 1 word, trigram = 3 words, **n-gram** = n words.)

### Image augmentation (for image GenAI)
Flip, rotate, crop, zoom, change brightness/colour, add blur.

### Simple synonym replacement code
```python
synonyms = {"happy": "glad", "good": "great", "big": "large"}

def augment(sentence):
    return " ".join(synonyms.get(w, w) for w in sentence.split())

print(augment("I am happy with this good product"))
# I am glad with this great product
```

> ⚠️ Augmentation must **keep the meaning (and label) the same**. "Not bad" → "Not good" changes the meaning — wrong!

---

## 5. Step 3 — Data Preparation

Raw text is messy. **Data preparation** makes it clean and consistent so the model learns the right things.

It has **4 layers**:

```
3.1 Cleanup  →  3.2 Basic Preprocessing  →  3.3 Optional Preprocessing  →  3.4 Advanced Preprocessing
```

### 3.1 🧹 Cleanup
Remove junk that doesn't carry meaning.

| Task | Example before | After |
|---|---|---|
| Remove HTML tags | `<p>Great phone!</p>` | `Great phone!` |
| Remove/convert emojis | `Loved it 😍` | `Loved it` or `Loved it [heart_eyes]` |
| Fix spelling | `Grate fone` | `Great phone` |
| Remove URLs | `see https://abc.com` | `see` |
| Remove extra spaces | `Hello     world` | `Hello world` |
| Remove duplicates | Same review twice | Keep one |

```python
import re

def clean(text):
    text = re.sub(r"<.*?>", "", text)          # remove HTML tags
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^\w\s]", "", text)        # remove emojis & special chars
    text = re.sub(r"\s+", " ", text).strip()   # extra spaces
    return text

print(clean("<p>Loved it 😍!!  Visit https://shop.com</p>"))
# Loved it Visit
```

🧠 **New term: Regex (Regular Expression)** — a mini-language for finding patterns in text, e.g. `r"http\S+"` means "the word http followed by non-space characters".

### 3.2 🔪 Basic Preprocessing — Tokenization
**Tokenization** = splitting text into smaller pieces (**tokens**).

| Type | Input | Output |
|---|---|---|
| **Sentence tokenization** | "I love AI. It is fun." | `["I love AI.", "It is fun."]` |
| **Word tokenization** | "I love AI" | `["I", "love", "AI"]` |
| **Subword tokenization** (used by LLMs) | "unhappiness" | `["un", "happi", "ness"]` |

```python
# pip install nltk
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

text = "I love AI. It is fun."
print(sent_tokenize(text))  # ['I love AI.', 'It is fun.']
print(word_tokenize(text))  # ['I', 'love', 'AI', '.', 'It', 'is', 'fun', '.']
```

🧠 **Why subword?** It handles new/rare words. Even if the model never saw "ChatGPTify", it can split it into known pieces.

### 3.3 🧰 Optional Preprocessing
Used **depending on the task**. (Classic ML often uses these; modern LLMs usually **don't** need most of them because they understand raw text.)

| Technique | Meaning | Example |
|---|---|---|
| **Lowercasing** | Make all text lowercase so "Apple" and "apple" match | "I Love AI" → "i love ai" |
| **Stop-word removal** | Remove very common words with little meaning | "this is a good movie" → "good movie" |
| **Punctuation removal** | Remove `. , ! ?` etc. | "Wow!!!" → "Wow" |
| **Stemming** | Chop word endings with simple rules (may create non-words) | "playing, played, plays" → "play"; "studies" → "studi" |
| **Lemmatization** | Convert to the real dictionary base form (smarter, slower) | "studies" → "study"; "better" → "good" |
| **Language detection** | Find which language the text is in | "Bonjour" → French |

🧠 **New term: Stop words** — super common words like *is, the, a, an, of, and*.
⚠️ Be careful: removing "**not**" turns "not good" into "good" — the opposite meaning! For sentiment tasks, keep negations.

🧠 **Stemming vs Lemmatization:**
| | Stemming | Lemmatization |
|---|---|---|
| Method | Cuts endings by rules | Uses a dictionary + grammar |
| "studies" | studi ❌ | study ✅ |
| Speed | Fast | Slower |
| Accuracy | Rough | Accurate |

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download("wordnet")

print(PorterStemmer().stem("studies"))              # studi
print(WordNetLemmatizer().lemmatize("studies"))     # study
```

```python
from nltk.corpus import stopwords
nltk.download("stopwords")
stops = set(stopwords.words("english"))
words = ["this", "is", "a", "good", "movie"]
print([w for w in words if w not in stops])          # ['good', 'movie']
```

### 3.4 🧪 Advanced Preprocessing
Adds **linguistic understanding**.

| Technique | What it does | Example |
|---|---|---|
| **POS tagging** (Part-of-Speech) | Labels each word's grammar role | "Ashish **runs** fast" → Ashish/NOUN runs/VERB fast/ADVERB |
| **Parsing** | Finds sentence structure — who did what to whom | "The dog chased the cat" → subject: dog, action: chased, object: cat |
| **Coreference resolution** | Figures out which words refer to the same thing | "**Ashish** went home because **he** was tired" → he = Ashish |
| **NER** (Named Entity Recognition) | Finds names, places, dates, orgs | "Accenture in Hyderabad" → ORG, LOCATION |

```python
# pip install spacy && python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Ashish works at Accenture in Hyderabad.")

for token in doc:
    print(token.text, token.pos_)     # POS tags
for ent in doc.ents:
    print(ent.text, ent.label_)       # Accenture ORG, Hyderabad GPE
```

---

## 6. Step 4 — Feature Engineering

Computers only understand **numbers**. **Feature engineering** (for text, also called **text vectorization** or **text representation**) = converting text into numbers the model can use.

🧠 **New term: Feature** — a measurable piece of information fed into a model. For a house: size, rooms. For text: word counts or embeddings.

### Techniques (from simple → modern)

Example sentences:
```
S1: "I love AI"
S2: "I love data"
Vocabulary (all unique words): [I, love, AI, data]
```

#### a) One-Hot Encoding
Each word = a vector with a single **1** at its position.
```
I    → [1,0,0,0]
love → [0,1,0,0]
AI   → [0,0,1,0]
data → [0,0,0,1]
```
❌ Huge vectors for big vocabularies; no meaning (AI and data look equally unrelated to everything).

#### b) Bag of Words (BoW)
Count how many times each word appears in a sentence (order ignored — like throwing words in a bag).
```
        I  love  AI  data
S1:  [  1,  1,   1,   0 ]
S2:  [  1,  1,   0,   1 ]
```
❌ Ignores word order and meaning.

#### c) TF-IDF (Term Frequency – Inverse Document Frequency)
Like BoW, but gives **more weight to rare, important words** and **less weight to words that appear everywhere**.
- **TF** = how often a word appears in this document.
- **IDF** = how rare the word is across all documents.
- Score = TF × IDF.

In our example, "I" and "love" appear in both sentences → low score; "AI" and "data" are unique → high score.

$$
\text{TF-IDF}(w) = \text{TF}(w) \times \log\left(\frac{N}{\text{number of documents containing } w}\right)
$$

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["I love AI", "I love data"]
print(CountVectorizer().fit_transform(docs).toarray())   # Bag of Words
print(TfidfVectorizer().fit_transform(docs).toarray())   # TF-IDF
```

#### d) Word Embeddings (Word2Vec, GloVe, FastText)
Each word → a **dense vector** (e.g., 100–300 numbers) that captures **meaning**. Similar words → similar vectors.
Famous example:
```
king − man + woman ≈ queen
```
```python
# pip install gensim
from gensim.models import Word2Vec
sentences = [["i", "love", "ai"], ["i", "love", "data"], ["ai", "needs", "data"]]
model = Word2Vec(sentences, vector_size=50, min_count=1)
print(model.wv["ai"][:5])          # first 5 numbers of the "ai" vector
```

#### e) Contextual / Transformer Embeddings (modern)
The **same word gets different vectors based on context**.
- "I sat on the river **bank**" vs "I deposited money in the **bank**" → different vectors.
Produced by models like BERT or sentence-transformers. These are what modern GenAI apps use.

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(["I love AI", "Artificial intelligence is great"])
print(vectors.shape)   # (2, 384) → 2 sentences, 384 numbers each
```

### Summary table

| Method | Captures meaning? | Captures context? | Vector size | Era |
|---|---|---|---|---|
| One-hot | ❌ | ❌ | Huge, sparse | Classic |
| Bag of Words | ❌ | ❌ | Huge, sparse | Classic |
| TF-IDF | Partly (importance) | ❌ | Huge, sparse | Classic |
| Word2Vec/GloVe | ✅ | ❌ | Small, dense | 2013+ |
| Transformer embeddings | ✅ | ✅ | Small, dense | Modern |

🧠 **New terms: Sparse vs Dense**
- **Sparse** = mostly zeros (e.g., `[0,0,0,1,0,0,...]`).
- **Dense** = mostly real numbers (e.g., `[0.21, -0.4, 0.88, ...]`) — compact and meaningful.

> 💡 **In LLM projects** you usually don't hand-build features. The LLM's own **tokenizer + embedding layer** does this automatically. But understanding these steps tells you *what's happening inside*.

---

## 7. Step 5 — Modeling

**Modeling** = choosing and training the model that does the actual work.

### Two main choices in GenAI

| Option | Examples | Pros | Cons |
|---|---|---|---|
| **Open-source LLM** | Llama, Mistral, Falcon (via Hugging Face) | Free to use, private, customizable | You need GPUs + you manage hosting |
| **Paid / closed LLM via API** | GPT, Claude, Gemini | Best quality, no infrastructure | Pay per token, data leaves your system |

### How you "model" in GenAI (from simple to heavy)

1. **Use as-is with prompts** (prompt engineering)
2. **RAG** — connect the LLM to your documents
3. **Fine-tune** — train further on your examples (often with LoRA)
4. **Train from scratch** — rare, very expensive

### Classic ML modeling (for comparison)
For tasks like spam detection, you might train **Logistic Regression**, **Naive Bayes**, or **Random Forest** on TF-IDF features. Fast and cheap — still very useful!

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts  = ["great product", "worst service", "loved it", "very bad"]
labels = [1, 0, 1, 0]               # 1 = positive, 0 = negative

vec = TfidfVectorizer()
X = vec.fit_transform(texts)
clf = LogisticRegression().fit(X, labels)

print(clf.predict(vec.transform(["really great"])))   # [1]
```

### 🧠 New terms
- **Train / Validation / Test split** — divide data into: **train** (learn), **validation** (tune settings), **test** (final unseen check). Common: 80/10/10.
- **Epoch** — one full pass through the training data.
- **Hyperparameters** — settings *you* choose before training (learning rate, epochs, batch size). Different from parameters, which the model learns.
- **Overfitting** — model memorizes training data and fails on new data. 🧒 Like a student who memorized answers but can't solve a new question.
- **Underfitting** — model is too simple to learn the pattern.

---

## 8. Step 6 — Evaluation

**Evaluation** = measuring how good the model is **before** giving it to users.

### Two kinds of evaluation

| Type | Meaning | When | Example |
|---|---|---|---|
| **Intrinsic evaluation** | Measured by technical metrics, usually by the ML team **before deployment** | During development | Accuracy, perplexity, BLEU |
| **Extrinsic evaluation** | Measured by **real-world/business results after deployment** | In production | Did customer complaints drop? Do users click 👍? |

### Common metrics

| Metric | Used for | Simple meaning |
|---|---|---|
| **Accuracy** | Classification | % of correct predictions |
| **Precision** | Classification | Of things predicted "spam", how many really were spam? |
| **Recall** | Classification | Of all actual spam, how much did we catch? |
| **F1-score** | Classification | Balance of precision and recall |
| **Perplexity** | Language models | How "surprised" the model is by real text. **Lower = better** |
| **BLEU** | Translation | Word overlap between model output and a reference translation |
| **ROUGE** | Summarization | Overlap between model summary and a reference summary |
| **Human evaluation** | Any GenAI | People rate helpfulness, correctness, tone |
| **LLM-as-a-judge** | Any GenAI | A strong LLM scores another model's answers with a rubric |
| **Faithfulness / Groundedness** | RAG | Is the answer supported by the retrieved documents? |

🧒 **Precision vs Recall example:** A fishing net.
- High **recall** = you catch almost all the fish (but maybe junk too).
- High **precision** = everything you caught is fish (but you may miss some).

```python
from sklearn.metrics import accuracy_score, classification_report
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]
print(accuracy_score(y_true, y_pred))          # 0.8
print(classification_report(y_true, y_pred))
```

> 💡 For GenAI there's often **no single correct answer**, so teams combine automatic metrics + human review + a **golden test set** (a fixed list of questions with expected answers they re-run after every change).

---

## 9. Step 7 — Deployment

**Deployment** = putting the model where real users/apps can use it.

### Typical deployment flow
```
Model  →  Wrap in an API (FastAPI/Flask)  →  Package in Docker  →  Host on cloud  →  Users call it
```

🧠 **New terms**
- **Endpoint** — a URL your app calls to get model answers, e.g. `POST /predict`.
- **Docker** — packs your code + dependencies into a **container** so it runs the same everywhere.
- **Serving** — running the model continuously to answer requests.
- **Batch vs real-time** — batch = process many records on a schedule (e.g., nightly summaries); real-time = answer instantly (chatbot).
- **Scaling** — adding more machines when traffic grows.

### Where to deploy
| Option | Examples |
|---|---|
| Cloud ML platforms | AWS SageMaker / Bedrock, Azure AI Foundry / Azure OpenAI, Google Vertex AI |
| Containers | Docker on ECS, EKS, Kubernetes |
| Serverless | AWS Lambda (light workloads, calling external LLM APIs) |
| Simple UIs | Streamlit, Gradio, Chainlit (quick demos) |
| Model servers | vLLM, Hugging Face TGI (for open-source LLMs on GPUs) |

### Minimal FastAPI example
```python
# pip install fastapi uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict(text: str):
    # here you'd call your model/LLM
    return {"input": text, "sentiment": "positive"}

# Run: uvicorn main:app --reload
# Open: http://127.0.0.1:8000/docs
```

---

## 10. Step 8 — Monitoring & Model Updating

Deployment is **not the end**. The world changes, so models get worse over time.

### What to monitor
| What | Why | Example |
|---|---|---|
| **Latency** | Users hate slow replies | Avg response 1.8s |
| **Errors / uptime** | Is it working? | 500 errors spiking |
| **Cost / token usage** | LLM bills grow fast | ₹ per 1,000 requests |
| **Quality** | Are answers still good? | 👍/👎 ratio dropping |
| **Hallucinations & safety** | Wrong or harmful outputs | Flagged responses |
| **Data drift** | Incoming data changed vs training data | New product names the model never saw |
| **Concept drift** | The meaning of "correct" changed | New company policy makes old answers wrong |

🧠 **New term: Drift** — when real-world data or meaning slowly changes, making the model less accurate. 🧒 Like a map that becomes outdated as new roads are built.

### Model updating (the loop)
```
Monitor → spot a problem → collect new data/feedback → retrain or update prompts/docs → evaluate → redeploy
```
In GenAI apps, "updating" often means **changing the prompt, refreshing the RAG documents, or switching to a newer model** — not always retraining.

---

## 11. Full Mini Example

**Project:** Auto-reply assistant for an online store's customer reviews.

| Step | What we do |
|---|---|
| 1. Acquisition | Pull 5,000 reviews from the database (SQL) |
| 2. Augmentation | Only 200 "angry" reviews → back-translate & paraphrase to make 600 |
| 3. Preparation | Remove HTML, emojis, URLs, duplicates; fix spelling |
| 4. Features | Create sentence embeddings for each review |
| 5. Modeling | Classify sentiment (small model) → LLM writes a polite reply based on sentiment |
| 6. Evaluation | Classifier F1 = 0.91; 3 humans rate 100 replies for politeness/correctness |
| 7. Deployment | FastAPI + Docker on cloud; endpoint `/reply` |
| 8. Monitoring | Track latency, cost, 👍/👎 from support staff; retrain monthly |

---

## 12. PRO Level: The Modern LLM App Pipeline (RAG)

Most real company GenAI apps today **don't train a model** — they use an existing LLM + their own documents. This is **RAG (Retrieval-Augmented Generation)**.

🧒 **Analogy:** An open-book exam. Instead of memorizing, the model **looks up the right pages** and then answers.

### Two phases

#### 🅰️ Ingestion (offline — prepare the knowledge)
```
Documents (PDF/Docs/Wiki) → Load → Clean → Chunk → Embed → Store in Vector DB
```

#### 🅱️ Query (online — answer the user)
```
User question → Embed question → Search Vector DB (find similar chunks)
             → Put top chunks + question into prompt → LLM → Answer (with citations)
```

🧠 **New terms**
- **Chunking** — splitting long documents into small pieces (e.g., 500 tokens with 50-token overlap) so the right part can be found and fit in the context window.
- **Vector database** — stores embeddings and quickly finds the most similar ones (Pinecone, ChromaDB, Weaviate, FAISS, pgvector, OpenSearch).
- **Similarity search** — finding vectors "closest" in meaning, often with **cosine similarity** (measures the angle between two vectors; closer to 1 = more similar).
- **Retriever** — the component that fetches relevant chunks.
- **Re-ranker** — a second model that re-orders retrieved chunks to put the best ones first.
- **Hybrid search** — combining keyword search (like BM25) with vector search for better results.

### Mini RAG example (ChromaDB)
```python
# pip install chromadb
import chromadb

client = chromadb.Client()
col = client.create_collection("hr_policies")

# Ingestion
col.add(
    documents=["Employees get 18 days of paid leave per year.",
               "Work from home is allowed 2 days a week with manager approval."],
    ids=["doc1", "doc2"]
)

# Query
question = "How many WFH days are allowed?"
result = col.query(query_texts=[question], n_results=1)
context = result["documents"][0][0]

prompt = f"Answer using ONLY this context:\n{context}\n\nQuestion: {question}"
print(prompt)   # send this prompt to any LLM
```

### Where the classic pipeline maps to RAG
| Classic step | In a RAG app |
|---|---|
| Data acquisition | Collect documents |
| Data preparation | Clean, remove headers/footers, chunk |
| Feature engineering | Create embeddings |
| Modeling | Choose LLM + prompt + retriever settings |
| Evaluation | Retrieval quality + answer faithfulness |
| Deployment | API/chat UI |
| Monitoring | Track bad answers, refresh documents |

### From RAG to Agents 🚀
An **agent** extends this pipeline: the LLM can **decide** to search documents, call APIs, run SQL, and loop until a goal is done. RAG often becomes **one of the agent's tools**.

---

## 13. PRO Level: LLMOps

**MLOps** = practices to build, deploy, and maintain ML models reliably (like DevOps for ML).
**LLMOps** = MLOps specialized for LLM apps.

| Area | What it covers | Example tools/ideas |
|---|---|---|
| **Prompt management** | Version and test prompts like code | Prompt templates in Git |
| **Experiment tracking** | Record which model/prompt/settings gave which results | MLflow, Weights & Biases |
| **Evaluation pipelines** | Auto-run golden test sets on every change | RAGAS, DeepEval, custom LLM-judge scripts |
| **Observability / Tracing** | See every step: prompt, retrieved chunks, tool calls, tokens, latency | LangSmith, Langfuse, OpenTelemetry |
| **Guardrails** | Block unsafe input/output, PII, prompt injection | Input/output filters, policy checks |
| **Cost control** | Caching, smaller models, token limits | Semantic cache, model routing |
| **CI/CD** | Automatic test + deploy | GitHub Actions, Jenkins |
| **Model registry** | Store approved model versions | MLflow Registry, SageMaker Registry |
| **Data governance** | Who can see which documents; lineage | Access controls on vector DB |

🧠 **New term: CI/CD** — Continuous Integration / Continuous Deployment: automatically testing and releasing code every time you change it.
🧠 **New term: PII** — Personally Identifiable Information (names, phone numbers, Aadhaar/PAN numbers) that must be protected.
🧠 **New term: Semantic cache** — saves answers to past questions; if a new question has the same *meaning*, return the saved answer instead of calling the LLM again (saves cost and time).

> 💡 As a **data engineer**, you're already strong in the ingestion, cleaning, orchestration, CI/CD, and governance parts — that's a big chunk of real GenAI work!

---

## 14. Common Mistakes

| Mistake | Why it hurts | Fix |
|---|---|---|
| Skipping data cleaning | Garbage in → garbage out | Always clean & deduplicate |
| Removing "not" as a stop word | Flips meaning | Keep negations |
| Testing on training data | Fake high scores | Keep a separate test set |
| Fine-tuning when RAG would do | Expensive, gets outdated | Try prompting → RAG first |
| Chunks too big or too small | Bad retrieval | Experiment with size + overlap |
| No monitoring after launch | Silent quality drop | Track quality, drift, cost |
| Sending sensitive data to external APIs | Privacy/compliance risk | Mask PII, use approved platforms |

---

## 15. Interview / Revision Cheat Sheet

**Pipeline order:**
> Acquisition → Augmentation → Preparation → Feature Engineering → Modeling → Evaluation → Deployment → Monitoring & Updating (loop)

- **No data?** Create it manually or generate **synthetic data** with an LLM.
- **Augmentation:** synonym replacement, bigram flip, back translation, adding noise, LLM paraphrasing.
- **Preparation layers:** Cleanup → Basic (tokenization) → Optional (lowercase, stop words, stemming, lemmatization, punctuation, language detection) → Advanced (POS, parsing, coreference, NER).
- **Stemming** = rough cut ("studi"); **Lemmatization** = real base word ("study").
- **Features:** One-hot → BoW → TF-IDF → Word2Vec → Transformer embeddings.
- **Modeling:** open-source (control) vs API (convenience); prompt → RAG → fine-tune → pre-train.
- **Intrinsic** evaluation = technical metrics before deploy; **Extrinsic** = business impact after deploy.
- **Perplexity** lower = better; **BLEU** = translation; **ROUGE** = summarization.
- **Deployment:** API (FastAPI) + Docker + cloud.
- **Monitoring:** latency, cost, quality, drift → retrain/update.
- **RAG:** Load → Chunk → Embed → Vector DB → Retrieve → Prompt → LLM.
- **LLMOps:** prompts versioning, evals, tracing, guardrails, cost control, CI/CD.

### ❓ Quick self-test
1. What would you do if you had zero training data?
2. Explain back translation with an example.
3. Why might removing stop words be dangerous for sentiment analysis?
4. What's the difference between TF-IDF and embeddings?
5. Intrinsic vs extrinsic evaluation — give one example each.
6. Draw the RAG pipeline from memory.

> Nail these and you understand the GenAI pipeline at a level most beginners never reach. 💪
