# Prompts Appendix

This appendix lists the main prompts used by the experimental pipeline.

## Query Expansion Prompt

Used by `src/augmented_retrieval/augmented_retrieval.py` with `gpt-4o-mini-2024-07-18`, temperature 0, max 200 tokens.

```text
Given the following question about past conversations, generate {n_expansions} alternative phrasings that preserve the original intent but use different wording.
Return ONLY a JSON array of strings, no other text.

Question: {question}
```

## Answer Generation Prompt

Used by `src/generation/run_generation.py` with retrieved LongMemEval context and `gpt-4o-mini-2024-07-18`.

```text
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.


History Chats:

{retrieved_history}

Current Date: {question_date}
Question: {question}
Answer:
```

When chain-of-thought mode is enabled, the answer template is:

```text
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.


History Chats:

{retrieved_history}

Current Date: {question_date}
Question: {question}
Answer (step by step):
```

## Context Note Extraction Prompt

Used by `src/generation/run_generation.py` when contextual note extraction (`--con true`) is enabled.

```text
I will give you a chat history between you and a user, as well as a question from the user. Write reading notes to extract all the relevant user information relevant to answering the answer. If no relevant information is found, just output "empty". 


Chat History:
Session Date: {session_date}
Session Content:
{session_content}

Question Date: {question_date}
Question: {question}
Extracted note (information relevant to answering the question):
```

## Phase 3 Classifier Router Prompt

Used by `src/augmented_retrieval/phase2_router.py` / `src/augmented_retrieval/phase3_classifier_router.py` with `gpt-4o-mini-2024-07-18`, temperature 0, max 16 tokens.

```text
You are a retrieval router for a conversational AI memory system.
Given a user's question, decide which retrieval strategy to use.

Strategies:
- DEFAULT: General retrieval with temporal weighting. Use for most questions, especially facts/counts and potentially updated or multi-conversation information.
- TEMPORAL: Enhanced temporal + query expansion. Use ONLY when the question explicitly asks when something happened, temporal ordering, or duration.
- PREFERENCE: Same retrieval config as TEMPORAL. Use for subjective user likes/dislikes, tastes, or personal preferences.
- ABSTENTION: Diversity-focused retrieval. Use ONLY when the question is likely asking about something never discussed.
- SIMPLE_FACT: Baseline retrieval. Use for direct single-fact user recall with no strong temporal/update/multi-session signal.

If unsure, choose DEFAULT.

Few-shot examples:
{fewshot_examples}

Question: "{question_text}"

Answer with ONLY one of: DEFAULT, TEMPORAL, PREFERENCE, ABSTENTION, SIMPLE_FACT
```

The few-shot examples used in the final run are saved in `results/phase3/summary_metrics.json` under `classifier.fewshot_examples`.

## QA Judge Prompts

Used by `src/evaluation/evaluate_qa.py` with `gpt-4o-2024-08-06`, temperature 0, max 10 tokens.

For single-session-user, single-session-assistant, and multi-session:

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. 

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

For temporal-reasoning:

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. 

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

For knowledge-update:

```text
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

For single-session-preference:

```text
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.
```

For abstention:

```text
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.
```

