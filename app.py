from flask import Flask, render_template, request, session, redirect, url_for
from jfk_qa_chatbot import search_faiss, answer_with_llm, aggregate_citations, USE_LLM_ANSWER

# Constants
SECRET_KEY = "your-secret-key"  # Replace with a secure key in production

# Initialize Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Global flag to track if session has been initialized
app.config['SESSION_INITIALIZED'] = False

@app.before_request
def clear_session_on_first_request():
    """
    Clear session data on the first request after app startup.
    """
    if not app.config.get('SESSION_INITIALIZED', False):
        session.clear()
        session['chat_history'] = []
        app.config['SESSION_INITIALIZED'] = True

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the main chat interface.
    - GET: Render the chat interface with chat history.
    - POST: Process user question, fetch results, and update chat history.
    """
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            top_results = search_faiss(question)
            top_chunks = [meta for meta, _ in top_results]

            if not top_results:
                answer = "No relevant results found."
                citations = ""
            elif USE_LLM_ANSWER:
                answer = answer_with_llm(question, top_chunks)
                citations = format_context_chunks_with_citations(aggregate_citations(top_chunks))
            else:
                answer = "\n\n".join(
                    f"[{i+1}] Pages {meta['start_file']}–{meta['end_file']} — File: {meta['file_name']}<br>{meta['chunk'][:500]}..."
                    for i, (meta, score) in enumerate(top_results)
                )
                citations = format_context_chunks_with_citations(aggregate_citations(top_chunks))

            session['chat_history'].append({
                "user": question,
                "bot": answer,
                "citations": citations
            })
            session.modified = True

            return redirect(url_for('index'))

    return render_template("index.html", chat_history=session['chat_history'])

@app.route("/clear", methods=["GET"])
def clear_chat():
    """
    Clear the chat history and redirect to the main page.
    """
    session['chat_history'] = []
    session.modified = True
    return redirect(url_for('index'))

def format_context_chunks_with_citations(chunks):
    """
    Format context chunks into an HTML unordered list of citations.

    Args:
        chunks (list or str): Context chunks or citation string.

    Returns:
        str: Formatted HTML string of citations.
    """
    if not chunks:
        return ""

    if isinstance(chunks, str):
        chunks = chunks.replace("Sources:", "").strip()
        citations = []
        current = ""
        in_brackets = False
        for char in chunks:
            if char == '[':
                in_brackets = True
            elif char == ']':
                in_brackets = False
            elif char == ',' and not in_brackets:
                citations.append(current.strip())
                current = ""
                continue
            current += char
        if current.strip():
            citations.append(current.strip())
    elif isinstance(chunks, list):
        citations = [chunk.replace("Sources:", "").strip() for chunk in chunks if chunk.strip()]
    else:
        return "Error: Invalid citation format."

    unique_citations = []
    seen_files = set()
    for citation in citations:
        if citation:
            file_name = citation.split(']')[0][1:] if ']' in citation else citation
            if file_name not in seen_files:
                seen_files.add(file_name)
                unique_citations.append(citation)

    if not unique_citations:
        return "No sources available."

    formatted_sources = "<div>Sources:</div><ul>" + "".join(
        f"<li>{citation}</li>" for citation in unique_citations
    ) + "</ul>"
    return formatted_sources

if __name__ == "__main__":
    # WARNING: Only use debug=True in development
    app.run(debug=False)