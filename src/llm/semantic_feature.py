"""
语义特征提取模块：为节点生成模拟推文并转换为Embedding
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, Optional, List
import pickle
import os

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, semantic features will use simple embeddings")

from .controller import LLMController
from .prompt_template import build_semantic_prompt

logger = logging.getLogger(__name__)


class SemanticFeatureExtractor:
    """
    语义特征提取器：使用LLM生成推文，然后转换为Embedding
    """
    
    def __init__(
        self,
        llm_controller: Optional[LLMController] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        """
        初始化语义特征提取器
        
        Args:
            llm_controller: LLM控制器实例（用于生成推文）
            embedding_model: 用于生成Embedding的模型
            device: 设备（"cuda" 或 "cpu"）
            cache_dir: 缓存目录（用于保存生成的推文和Embedding）
        """
        self.llm_controller = llm_controller or LLMController(use_mock=True)
        self.embedding_model_name = embedding_model
        self.device = device
        self.cache_dir = cache_dir
        
        self.embedding_tokenizer = None
        self.embedding_model = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model {embedding_model}...")
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
                self.embedding_model = AutoModel.from_pretrained(embedding_model)
                if device == "cuda":
                    self.embedding_model = self.embedding_model.to(device)
                self.embedding_model.eval()
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        else:
            logger.warning("transformers not available, using simple hash-based embeddings")
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        将文本转换为Embedding向量
        
        Args:
            text: 输入文本
        
        Returns:
            embedding: Embedding向量（归一化）
        """
        if self.embedding_model is None or self.embedding_tokenizer is None:
            # 简单fallback：使用hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # 生成128维向量
            np.random.seed(hash_int % (2**32))
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        
        try:
            inputs = self.embedding_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # 使用[CLS] token或平均池化
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output.cpu().numpy()[0]
                else:
                    # 平均池化
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
            # 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            np.random.seed(hash_int % (2**32))
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def generate_tweet_for_node(self, node_id: int, graph: nx.Graph) -> str:
        """
        为节点生成模拟推文
        
        Args:
            node_id: 节点ID
            graph: 图对象
        
        Returns:
            tweet: 生成的推文文本
        """
        if node_id not in graph:
            return f"User {node_id} has no content."
        
        opinion = graph.nodes[node_id].get('opinion', 0.0)
        degree = graph.degree(node_id)
        
        # 构建Prompt
        prompt = build_semantic_prompt(node_id, opinion, degree)
        
        # 调用LLM生成推文
        if hasattr(self.llm_controller, '_call_llm'):
            response = self.llm_controller._call_llm(prompt)
        else:
            # Mock模式：生成简单的模拟推文
            if opinion < -0.5:
                response = f"I strongly disagree with this perspective. We need to reconsider our approach. #{node_id}"
            elif opinion < 0:
                response = f"I have some concerns about this. Let's discuss. #{node_id}"
            elif opinion < 0.5:
                response = f"This seems reasonable. I support this direction. #{node_id}"
            else:
                response = f"I fully support this! This is the right way forward. #{node_id}"
        
        # 清理响应（移除可能的解释文字）
        tweet = response.strip()
        # 如果响应太长，截断
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        
        return tweet
    
    def extract_semantic_embedding(
        self, 
        node_id: int, 
        graph: nx.Graph,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        提取节点的语义Embedding
        
        Args:
            node_id: 节点ID
            graph: 图对象
            use_cache: 是否使用缓存
        
        Returns:
            embedding: 语义Embedding向量
        """
        # 检查缓存
        if use_cache and self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"embeddings_{node_id}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        return cached_data['embedding']
                except Exception as e:
                    logger.warning(f"Failed to load cache for node {node_id}: {e}")
        
        # 生成推文
        tweet = self.generate_tweet_for_node(node_id, graph)
        
        # 转换为Embedding
        embedding = self._text_to_embedding(tweet)
        
        # 保存缓存
        if use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"embeddings_{node_id}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'tweet': tweet, 'embedding': embedding}, f)
            except Exception as e:
                logger.warning(f"Failed to save cache for node {node_id}: {e}")
        
        return embedding
    
    def batch_extract_embeddings(
        self,
        graph: nx.Graph,
        node_ids: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        批量提取多个节点的语义Embedding
        
        Args:
            graph: 图对象
            node_ids: 节点ID列表（如果为None，则处理所有节点）
            use_cache: 是否使用缓存
        
        Returns:
            embeddings: 字典，键为节点ID，值为Embedding向量
        """
        if node_ids is None:
            node_ids = list(graph.nodes())
        
        embeddings = {}
        for node_id in node_ids:
            try:
                embeddings[node_id] = self.extract_semantic_embedding(
                    node_id, graph, use_cache=use_cache
                )
            except Exception as e:
                logger.error(f"Failed to extract embedding for node {node_id}: {e}")
                # 使用零向量作为fallback
                embeddings[node_id] = np.zeros(128, dtype=np.float32)
        
        return embeddings


# 便捷函数
def extract_semantic_embedding(node_id: int, graph: nx.Graph) -> np.ndarray:
    """
    便捷函数：提取单个节点的语义Embedding
    
    Args:
        node_id: 节点ID
        graph: 图对象
    
    Returns:
        embedding: 语义Embedding向量
    """
    extractor = SemanticFeatureExtractor()
    return extractor.extract_semantic_embedding(node_id, graph)


def generate_tweet_for_node(node_id: int, graph: nx.Graph) -> str:
    """
    便捷函数：为节点生成模拟推文
    
    Args:
        node_id: 节点ID
        graph: 图对象
    
    Returns:
        tweet: 生成的推文文本
    """
    extractor = SemanticFeatureExtractor()
    return extractor.generate_tweet_for_node(node_id, graph)
