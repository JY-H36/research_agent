"""
查询改写模块
将用户的科研问题拆解为多条同义/多角度的检索查询，解决学术术语不统一的问题
"""
import json
import logging
from typing import List

from config import QUERY_VARIANTS
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 查询改写 Prompt
# ============================================================
REWRITE_PROMPT = """You are an academic search expert with broad knowledge across all research fields. Your task is to expand a single research question into {N} diverse search queries, improving recall in academic databases where different papers use different terminology for the same concepts.

## Rewriting Strategy

For the given question, think through these steps before writing:

1. **Identify core concepts**: Extract 2–4 key concepts from the question (e.g., a task, a method, a model, a dataset, a domain).

2. **Diversify each concept**: For each core concept, use your broad academic knowledge to list alternative terms that different research communities might use. Consider:
   - Full names vs. abbreviations vs. code names for models/methods
   - Functional descriptions as alternatives to brand names (e.g., a specific model can also be described by what it does)
   - Different academic communities' preferred terminology for the same idea
   - Broader and narrower terms (hypernyms and hyponyms)

3. **Vary the angle**: Each query should emphasize a different combination of concepts or a different perspective:
   - One query focused on the method/architecture
   - One query focused on the task/problem formulation
   - One query using broader conceptual terms rather than specific brand names
   - One query closer to the user's original phrasing but with key terms swapped
   - If the input contains Chinese, include one Chinese-language query

4. **Keep queries concise**: Each query should be a keyword phrase (≤20 words), not a full question. Remove filler words like "有没有", "如何", "system", "method", "research on".

## Output

Return ONLY a JSON array of exactly {N} strings. No explanation, no markdown, just the raw JSON array.

User question: {question}

JSON array:"""


def rewrite_query(question: str, num_variants: int = QUERY_VARIANTS) -> List[str]:
    """
    将用户问题改写为多条检索查询
    返回: 查询字符串列表（如果 LLM 失败则返回 [原始问题]）
    """
    if not question or not question.strip():
        return [question]

    prompt = REWRITE_PROMPT.format(N=num_variants, question=question)

    try:
        from agent.llm_service import chat_completion, extract_content

        logger.debug("查询改写: 原始问题='%s', 目标变体数=%d", question[:120], num_variants)
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.4,  # 适度随机性，保证多样性
            max_tokens=512,
        )
        content = extract_content(response)

        # 解析 JSON 数组
        queries = _parse_queries(content, num_variants)

        if queries:
            logger.info("查询改写: %d 个变体 → %s", len(queries),
                       [q[:60] + "..." for q in queries])
            # 确保原始问题相关词汇被覆盖（把原问题也加进去）
            if question not in queries:
                queries.insert(0, question[:200])
            return queries[:num_variants + 1]  # +1 是因为加了原问题
        else:
            raise ValueError("解析结果为空")

    except Exception as e:
        logger.warning("查询改写失败 (%s)，使用原始查询 + 规则后备", e)
        return _fallback_rewrite(question, num_variants)


def _parse_queries(content: str, expected_count: int) -> List[str]:
    """从 LLM 响应中解析查询列表"""
    # 尝试直接解析 JSON 数组
    content = content.strip()
    # 去掉可能的 markdown 代码块标记
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        queries = json.loads(content)
        if isinstance(queries, list) and len(queries) > 0:
            return [q.strip() for q in queries if q and q.strip()]
    except json.JSONDecodeError:
        pass

    # JSON 解析失败，尝试按行分割
    lines = [l.strip().lstrip('0123456789.-) "') for l in content.split("\n") if l.strip()]
    return [l for l in lines if len(l) > 5][:expected_count]


def _fallback_rewrite(question: str, count: int) -> List[str]:
    """
    规则后备：不用 LLM，通过简单规则生成变体
    至少保证原问题被检索
    """
    queries = [question[:200]]

    # 简单替换规则生成 2~3 个额外变体
    replacements = [
        ("spoofing", "fake"),
        ("fake", "forgery"),
        ("detection", "localization"),
        ("伪造", "虚假"),
        ("检测", "识别"),
    ]

    for old, new in replacements:
        if old.lower() in question.lower():
            new_q = question.replace(old, new).replace(old.capitalize(), new.capitalize())
            if new_q not in queries:
                queries.append(new_q[:200])
        if len(queries) >= count:
            break

    logger.debug("规则后备改写: %d 个变体", len(queries))
    return queries
