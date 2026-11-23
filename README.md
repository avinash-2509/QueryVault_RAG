# Simple RAG with LangChain and Streamlit

A small retrieval-augmented generation app that lets you upload PDF, Markdown,
or text files and ask questions about their contents.

## What it uses

- Streamlit for the chat UI
- LangChain for document splitting, embeddings, retrieval, prompting, and model calls
- OpenAI `text-embedding-3-small` embeddings
- An editable OpenAI chat-model setting (defaults to `gpt-6-luna`)
- Chroma as a persistent local vector database

## Run locally

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Either enter your OpenAI API key in the app, or copy `.env.example` to `.env`
   and add your key.

4. Start Streamlit:

   ```powershell
   python -m streamlit run app.py
   ```

5. Upload one or more files, select **Process documents**, then ask questions.

## Notes

- Scanned PDFs need OCR before this app can read them.
- Vectors persist in `.rag_data/chroma` and reload automatically after app restarts.
- Chat history still lives only in the current Streamlit session.
- Re-process the documents after changing the selected files or chunk settings.
- OpenAI API usage incurs normal API charges.
