"""
LLM控制器：使用Qwen模型进行推理，生成干预策略权重（沟通成本）
"""

import re
import logging
from typing import Optional, Dict, Any
import numpy as np
import networkx as nx

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, LLMController will use mock mode")

from .prompt_template import build_intervention_prompt

logger = logging.getLogger(__name__)


class LLMController:
    """
    LLM控制器，用于生成干预策略权重（沟通成本）
    
    支持两种模式：
    1. 真实模式：使用HuggingFace transformers加载Qwen模型
    2. Mock模式：使用启发式规则（用于测试或没有GPU的情况）
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        device: str = "cuda",
        use_mock: bool = False,
        temperature: float = 0.1
    ):
        """
        初始化LLM控制器
        
        Args:
            model_name: 模型名称或路径（HuggingFace格式）
            device: 设备（"cuda" 或 "cpu"）
            use_mock: 是否使用Mock模式（不加载真实模型）
            temperature: 生成温度（越低越确定）
        """
        self.model_name = model_name
        self.device = device
        self.use_mock = use_mock
        self.temperature = temperature
        
        self.tokenizer = None
        self.model = None
        
        if not use_mock and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading model {model_name} on {device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device if device == "cuda" else None,
                    trust_remote_code=True
                )
                if device == "cpu":
                    self.model = self.model.to(device)
                self.model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Falling back to mock mode.")
                self.use_mock = True
        else:
            if use_mock:
                logger.info("Using mock mode (no LLM loaded)")
            else:
                logger.warning("transformers not available, using mock mode")
                self.use_mock = True
    
    def _extract_subgraph_info(self, node_id: int, graph: nx.Graph) -> str:
        """
        提取目标节点及其邻居的文本描述
        
        Args:
            node_id: 目标节点ID
            graph: 图对象
        
        Returns:
            subgraph_text: 子图的文本描述
        """
        if node_id not in graph:
            return f"Node {node_id} does not exist in the graph."
        
        node_opinion = graph.nodes[node_id].get('opinion', 0.0)
        neighbors = list(graph.neighbors(node_id))
        n_neighbors = len(neighbors)
        
        if n_neighbors == 0:
            return f"Node {node_id} has an opinion of {node_opinion:.2f} and is isolated (no neighbors)."
        
        # 计算邻居的平均观点值
        neighbor_opinions = [graph.nodes[n].get('opinion', 0.0) for n in neighbors]
        avg_neighbor_opinion = np.mean(neighbor_opinions)
        
        # 计算观点分歧（与邻居的平均差异）
        opinion_disagreement = abs(node_opinion - avg_neighbor_opinion)
        
        # 判断观点是否极端
        is_extreme = abs(node_opinion) > 0.7
        extreme_label = "extreme" if is_extreme else "moderate"
        
        # 判断邻居是否持相反观点
        opposing_count = sum(1 for op in neighbor_opinions if op * node_opinion < 0)
        opposing_ratio = opposing_count / n_neighbors if n_neighbors > 0 else 0
        
        subgraph_text = f"""Node {node_id} has an {extreme_label} opinion ({node_opinion:.2f}).
It connects to {n_neighbors} neighbors with an average opinion of {avg_neighbor_opinion:.2f}.
The opinion disagreement with neighbors is {opinion_disagreement:.2f}.
{opposing_count} out of {n_neighbors} neighbors ({opposing_ratio*100:.1f}%) hold opposing views."""
        
        return subgraph_text
    
    def _mock_intervention_weight(self, node_id: int, graph: nx.Graph) -> float:
        """
        Mock 模式下的启发式权重计算（不调用 LLM）
        
        基于节点特征计算干预权重：
        - 观点越极端，需要的干预权重越大
        - 如果邻居观点相反，需要更多干预
        
        Args:
            node_id: 目标节点索引
            graph: 图对象
        
        Returns:
            weight: 干预权重（0.0 到 1.0）
        """
        if node_id not in graph:
            return 0.5
        
        node_opinion = graph.nodes[node_id].get('opinion', 0.0)
        neighbors = list(graph.neighbors(node_id))
        
        # 规则1: 观点越极端（绝对值越大），需要的干预权重越大
        base_weight = abs(node_opinion) * 0.7  # 0.0 到 0.7
        
        # 规则2: 如果邻居观点相反，需要更多干预
        if len(neighbors) > 0:
            neighbor_opinions = [graph.nodes[n].get('opinion', 0.0) for n in neighbors]
            opposing_count = sum(1 for op in neighbor_opinions if op * node_opinion < 0)
            opposing_ratio = opposing_count / len(neighbors)
            base_weight += opposing_ratio * 0.3  # 额外 0.0 到 0.3
        
        # 规则3: 如果节点孤立，需要较少干预
        if len(neighbors) == 0:
            base_weight *= 0.5
        
        # 限制在 0.0-1.0 之间
        weight = max(0.0, min(1.0, base_weight))
        
        # 确保不会太小（至少 0.1）
        weight = max(0.1, weight)
        
        return weight
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM生成回复
        
        Args:
            prompt: 输入的Prompt
        
        Returns:
            response: LLM的回复文本
        """
        if self.use_mock:
            # Mock模式：返回固定值（实际权重在 get_intervention_weight 中计算）
            return "0.5"
        
        try:
            # 构建输入
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 使用tokenizer格式化（Qwen2格式）
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=50,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "0.5"  # 默认返回中等成本
    
    def _parse_float_from_response(self, response: str) -> float:
        """
        从LLM回复中解析浮点数
        
        Args:
            response: LLM的回复文本
        
        Returns:
            cost: 沟通成本（0.0到1.0之间的浮点数）
        """
        # 尝试提取浮点数
        patterns = [
            r'\b0?\.\d+\b',  # .5, 0.5
            r'\b[01]\.\d+\b',  # 0.5, 1.0
            r'\b\d+\.\d+\b',  # 任何浮点数
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    value = float(matches[0])
                    # 限制在0.0到1.0之间
                    value = max(0.0, min(1.0, value))
                    return value
                except ValueError:
                    continue
        
        # 如果无法解析，返回默认值
        logger.warning(f"Could not parse float from response: {response}. Using default 0.5")
        return 0.5
    
    def get_intervention_weight(self, node_id: int, graph: nx.Graph) -> float:
        """
        根据节点及其邻居信息，生成干预权重（沟通成本）
        
        Args:
            node_id: 目标节点索引
            graph: 图对象
        
        Returns:
            weight: 干预权重（0.0 到 1.0 之间的浮点数）
        """
        # 如果是 Mock 模式，直接使用启发式规则计算（不调用 LLM）
        if self.use_mock:
            weight = self._mock_intervention_weight(node_id, graph)
            logger.debug(f"Node {node_id}: intervention weight (mock) = {weight:.3f}")
            return weight
        
        # 真实模式：使用 LLM
        # 1. 提取子图特征
        subgraph_text = self._extract_subgraph_info(node_id, graph)
        
        # 2. 构建Prompt
        prompt = build_intervention_prompt(subgraph_text)
        
        # 3. 调用LLM
        response = self._call_llm(prompt)
        
        # 4. 解析输出
        weight = self._parse_float_from_response(response)
        
        logger.debug(f"Node {node_id}: intervention weight (LLM) = {weight:.3f}")
        
        return weight
    
    def get_intervention_strategy(self, subgraph_text: str) -> float:
        """
        根据子图文本描述直接获取干预策略（兼容原始接口）
        
        Args:
            subgraph_text: 子图的文本描述
        
        Returns:
            cost: 沟通成本（0.0 到 1.0 之间的浮点数）
        """
        prompt = build_intervention_prompt(subgraph_text)
        response = self._call_llm(prompt)
        cost = self._parse_float_from_response(response)
        return cost
