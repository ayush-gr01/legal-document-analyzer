from langchain_core.prompts import PromptTemplate

# System message for the LLM – includes disclaimer and instruction set.
SYSTEM_PROMPT = (
    "You are a legal‑document explanation assistant. "
    "Answer the user's question *only* using the provided context passages. "
    "Do not fabricate information, dates, amounts, obligations, or legal rules. "
    "If the answer cannot be found in the context, respond with:"
    "\n\n" "I could not find sufficient information in the uploaded documents to answer this question."
    "\n\n"
    "When you cite a passage, include the source filename and page number in the format "
    "[source.pdf - page X]. "
    "Do not mention the internal chunk index. "
    "Finally, add a short disclaimer that the response is informational and not legal advice."
)

# Prompt template that will be filled with the question and retrieved passages.
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "{system_prompt}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"  # The LLM will generate the answer after this.
    ),
)

def build_prompt(question: str, contexts: list[str]) -> str:
    """Create the final prompt string given a question and a list of context passages.

    Each context passage will be prefixed with its source citation.
    """
    # Join passages with a separator for readability.
    context_text = "\n---\n".join(contexts)
    return PROMPT_TEMPLATE.format(question=question, context=context_text, system_prompt=SYSTEM_PROMPT)
