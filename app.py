from flask import Flask, render_template, request
from rag import RAG

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            answer = RAG.run(question)
        else:
            answer = "Please enter a question."

    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)