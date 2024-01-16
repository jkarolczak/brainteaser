from textwrap import dedent

import streamlit as st

from solvers import ZeroShotGPT, ZeroShotWithReasoningGPT, InContextGPT
from structs import Instance

solvers_dict = {
    "Vanilla gpt-3.5-turbo": ZeroShotGPT("gpt-3.5-turbo"),
    "One-Shot gpt-3.5-turbo for sentence puzzle": ZeroShotGPT("ft:gpt-3.5-turbo-0613:ncodex::8QhGOWvR"),
    "One-Shot gpt-3.5-turbo for word puzzle": ZeroShotGPT("ft:gpt-3.5-turbo-0613:ncodex::8R1R0Vi0"),
    "InContext gpt-3 for sentence puzzle": InContextGPT(InContextGPT.Context.SENTENCE, "gpt-3.5-turbo"),
    "InContext gpt-3.5-turbo for word puzzle": InContextGPT(InContextGPT.Context.WORD, "gpt-3.5-turbo"),
    "Reasoning gpt-3.5-turbo for word puzzle": ZeroShotWithReasoningGPT("gpt-4-1106-preview"),
    "Vanilla gpt-4": ZeroShotGPT("gpt-4-1106-preview"),
    "InContext gpt-4 for sentence puzzle": InContextGPT(InContextGPT.Context.SENTENCE, "gpt-4-1106-preview"),
    "InContext gpt-4 for word puzzle": InContextGPT(InContextGPT.Context.WORD, "gpt-4-1106-preview"),
    "Reasoning gpt-4 for word puzzle": ZeroShotWithReasoningGPT("gpt-4-1106-preview"),
}


def format_answer(question: str, choices: list[str], solver_name: str, answer_idx: int) -> str:
    answer_letter = "ABCD"[answer_idx]
    answer_content = choices[answer_idx]
    return (f"According to *{solver_name}*, the correct answer to question \"{question}\" is the answer **{answer_letter}) "
            f"{answer_content}**")


def generate_answer(question: str, choices: list[str], solver_name: str) -> str:
    instance = Instance(question=question, choice_list=choices)
    solver = solvers_dict[solver_name]
    answer_idx = solver.solve_instance(instance)
    answer_meassage = format_answer(question, choices, solver_name, answer_idx)
    return answer_meassage


def main():
    st.set_page_config(
        page_title="Brainteaser solver",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(dedent("""\
    # Brainteasers solver
    This is an app prototype that allows testing question-answering systems implemented by 
    [@annprzy](https://github.com/annprzy), [@wtainser](https://github.com/wtaisner), and 
    [@jkarolczak](https://github.com/jkarolczak) as a part of SemeEal2024 task 9: A Novel Task Defying Common Sense. 
    Brainteaser is a Question Answering task challenging models to exhibit creative, lateral thinking in contrast to 
    traditional logical reasoning. It includes Sentence and Word Puzzles, requiring models to defy commonsense defaults and 
    think unconventionally. For details see [contest's home page](https://brainteasersem.github.io).
    """))
    col1, col2 = st.columns(2)
    col1.markdown("## Brainteaser")
    col1_form = col1.form("brainteaser")

    col2.markdown("## Answer")
    col2_answer = col2.info("To generate an answer fill the form on the left and press the submit button.")

    with col1_form:
        st.markdown("Question")
        question = st.text_area(label="question", label_visibility="collapsed", placeholder="Put a question here")
        st.markdown("Choices")
        choice_list = [
            st.text_input(label=f"Choice {choice}", label_visibility="collapsed", placeholder=f"Choice {choice}")
            for choice in "ABCD"]
        st.markdown("Solver")
        solver_name = st.selectbox(label="Solver", options=solvers_dict.keys(), label_visibility="collapsed",
                                   help="Select model to solver the puzzle")
        if st.form_submit_button():
            if all([question] + choice_list):
                answer = generate_answer(question, choice_list, solver_name)
                col2_answer.success(answer)
            else:
                col2_answer.error("All fields (question and four answers) in the question form have to be filled!")


if __name__ == "__main__":
    main()
