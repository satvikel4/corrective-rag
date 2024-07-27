from langsmith import Client
from langchain import hub
from langchain_openai import ChatOpenAI

def predict_custom_agent_answer(example: dict, workflow):
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = workflow.invoke(
        {"question": example["input"], "steps": []}, config
    )
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}

def create_dataset():
    client = Client()
    examples = [
        ("How does the ReAct agent use self-reflection? ", "ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."),
        ("What are the types of biases that can arise with few-shot prompting?", "The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."),
        ("What are five types of adversarial attacks?", "Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."),
        ("Who did the Chicago Bears draft first in the 2024 NFL draft?", "The Chicago Bears drafted Caleb Williams first in the 2024 NFL draft."),
        ("Who won the 2024 NBA finals?", "The Boston Celtics on the 2024 NBA finals"),
    ]

    dataset_name = "Corrective RAG Agent Testing"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        inputs, outputs = zip(*[({"input": text}, {"output": label}) for text, label in examples])
        client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

def run_evaluation():
    from langsmith.evaluation import evaluate
    
    dataset_name = "Corrective RAG Agent Testing"
    model_tested = "llama3.1"
    metadata = "llama3.1"
    experiment_prefix = f"custom-agent-{model_tested}"
    experiment_results = evaluate(
        predict_custom_agent_answer,
        data=dataset_name,
        evaluators=[answer_evaluator, check_trajectory_custom],
        experiment_prefix=experiment_prefix + "-answer-and-tool-use",
        num_repetitions=3,
        max_concurrency=1,
        metadata={"version": metadata},
    )

def answer_evaluator(run, example):
    grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    answer_grader = grade_prompt_answer_accuracy | llm
    score = answer_grader.invoke({
        "question": input_question,
        "correct_answer": reference,
        "student_answer": prediction,
    })
    return {"key": "answer_v_reference_score", "score": score["Score"]}

def check_trajectory_custom(root_run, example):
    expected_trajectory_1 = ["retrieve_documents", "grade_document_retrieval", "web_search", "generate_answer"]
    expected_trajectory_2 = ["retrieve_documents", "grade_document_retrieval", "generate_answer"]
    tool_calls = root_run.outputs["steps"]
    score = 1 if tool_calls in [expected_trajectory_1, expected_trajectory_2] else 0
    return {"score": int(score), "key": "tool_calls_in_exact_order"}