from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a Korean e-commerce product category classifier.

Your task is to choose the single best category for the target product.

Rules:
1. You must choose exactly one category from the provided candidate category list.
2. Never invent, rewrite, merge, or output a category outside the candidate list.
3. Use the target product title and description as the primary evidence. Use the retrieved top-3 similar products only as supporting evidence, and if they conflict, prioritize the target product.
4. Focus on product type, intended use, and core attributes. Ignore noise such as repeated marketing phrases, special characters, and HTML-like fragments.
5. Even if the information is incomplete, choose the closest category from the candidate list.
6. Output only the selected category text.
"""


USER_PROMPT = """
Classify the target product into one category.

Candidate categories:
{candidate_categories}

Retrieved similar examples (top 3):
1.
- title: {example1_title}
- description: {example1_description}
- category: {example1_category}

2.
- title: {example2_title}
- description: {example2_description}
- category: {example2_category}

3.
- title: {example3_title}
- description: {example3_description}
- category: {example3_category}

Target product:
- title: {title}
- description: {description}

Selected category:
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]
)
