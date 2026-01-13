"""
Prompt模板模块：用于构建LLM推理的Prompt
"""

def build_intervention_prompt(subgraph_text: str) -> str:
    """
    构建干预策略的Prompt，用于评估沟通成本
    
    Args:
        subgraph_text: 子图的文本描述（目标节点及其邻居的信息）
    
    Returns:
        prompt: 完整的Prompt字符串
    """
    prompt = f"""You are a social network mediator analyzing communication costs.

{subgraph_text}

Task: Determine the communication cost (0.0 to 1.0) required to moderate this node.
- 0.0 means very low cost (easy to communicate and persuade)
- 1.0 means very high cost (difficult to communicate and persuade)

Consider factors:
1. How extreme is the node's opinion?
2. How many neighbors have opposing views?
3. How strong are the connections?
4. How polarized is the local network?

Output only a single float number between 0.0 and 1.0, nothing else."""
    
    return prompt


def build_semantic_prompt(node_id: int, opinion: float, degree: int) -> str:
    """
    构建生成模拟推文的Prompt
    
    Args:
        node_id: 节点ID
        opinion: 节点观点值（-1到1）
        degree: 节点度数
    
    Returns:
        prompt: 完整的Prompt字符串
    """
    opinion_label = "very negative" if opinion < -0.5 else "negative" if opinion < 0 else "positive" if opinion < 0.5 else "very positive"
    
    prompt = f"""Generate a realistic social media post (tweet-like) for a user in a social network.

User characteristics:
- User ID: {node_id}
- Opinion stance: {opinion_label} (value: {opinion:.2f})
- Network connections: {degree} neighbors

The post should:
1. Reflect the user's opinion stance naturally
2. Be concise (within 280 characters)
3. Sound like a real social media post
4. Be contextually appropriate

Generate only the post content, no explanations."""
    
    return prompt
