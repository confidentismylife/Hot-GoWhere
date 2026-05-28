"""Lightweight RAG knowledge base for disaster evacuation domain knowledge.

Uses ChromaDB with a small embedding model (bge-small-zh).
Pre-populated with essential evacuation safety rules across disaster types.
"""

import json
import os
from typing import List

# Pre-compiled disaster knowledge (avoids needing large training corpus at startup)
CORE_KNOWLEDGE = {
    "fire": [
        "火灾疏散原则: 弯腰低姿前进,烟雾向上聚集,地面空气相对清洁。用湿布捂住口鼻可过滤部分有毒气体。",
        "火灾时切勿乘坐电梯,应走楼梯。电梯可能因断电卡住,且电梯井会成为烟囱效应通道。",
        "如果门把手很烫,说明门外有火,不要开门。用湿布堵住门缝,在窗口等待救援。",
        "商场/地铁火灾: 注意地面疏散指示灯,沿指示方向撤离。不要往高处跑,火势和烟雾向上蔓延。",
        "身上着火时: 停住、倒地、翻滚(Stop-Drop-Roll)。不要奔跑,奔跑会加速燃烧。",
        "火灾中最危险的是烟雾而非火焰。多数火灾遇难者是吸入有毒烟雾窒息。能见度低于3米时爬行前进。",
        "选择疏散出口时优先选择防火楼梯间,其次是有自然通风的出口。避免通过火源上风方向的通道。",
        "帮助老人和儿童时要量力而行。如果烟雾太浓,自己先保证安全,找到救援人员后再返回帮助。",
    ],
    "earthquake": [
        "地震时首先: 趴下、掩护、抓牢(Drop-Cover-Hold)。远离窗户、吊灯、高大柜子。",
        "地震暂停后立即撤离。走楼梯不乘电梯。注意余震,移动时保护头部。",
        "室外疏散: 远离建筑物、电线杆、广告牌。前往开阔地带如广场、公园。",
        "地下空间(地铁/地下商场)地震: 保持冷静,按应急灯指示撤离。结构可能受损,注意头顶坠落物。",
        "如果被困: 不要使用打火机(可能有燃气泄漏)。敲击管道发出求救信号。保存体力。",
    ],
    "flood": [
        "洪水向高处转移,不要进入地下空间。上楼而非下楼。",
        "不要涉水通过流动的水,15cm深的流水就能把人冲倒。",
        "地下车库/地铁进水: 立即弃车逃生,水压会使车门无法打开。",
        "注意触电风险。避开水中倒下的电线杆和电器设备。",
    ],
    "general": [
        "任何灾害中保持冷静是最重要的。恐慌会导致错误判断,如盲目跟随人群或选择错误出口。",
        "熟悉环境的多个出口位置可以在紧急情况下节省宝贵时间。进入陌生场所时先观察疏散图。",
        "听从现场工作人员和应急救援人员的指挥。他们了解建筑结构和最佳疏散路线。",
        "帮助弱势群体(老人、儿童、残疾人)可以提高整体疏散效率。组织有序疏散比各自逃跑更有效。",
        "手机可以作为手电筒使用。提前下载离线地图在信号中断时仍然有用。",
    ],
}


class DisasterKnowledgeBase:
    """In-memory knowledge base with ChromaDB persistence option."""

    def __init__(self, persist_dir: str = None):
        self._docs = []
        self._doc_index = {}  # disaster_type -> [doc_indices]

        self._load_core_knowledge()

        # Try to initialize ChromaDB for larger-scale usage
        self._chroma = None
        self._encoder = None
        if persist_dir:
            self._init_chroma(persist_dir)

    def _load_core_knowledge(self):
        """Load the pre-compiled core knowledge."""
        idx = 0
        for dtype, docs in CORE_KNOWLEDGE.items():
            self._doc_index[dtype] = []
            for doc in docs:
                self._docs.append(doc)
                self._doc_index[dtype].append(idx)
                idx += 1

    def _init_chroma(self, persist_dir: str):
        """Lazy-init ChromaDB with sentence-transformer embeddings."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._chroma = chromadb.PersistentClient(path=persist_dir)
            collection_name = "disaster_evacuation"

            # Check if collection exists
            try:
                self._collection = self._chroma.get_collection(collection_name)
            except Exception:
                self._collection = self._chroma.create_collection(collection_name)

            self._encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

            # Populate if empty
            if self._collection.count() == 0:
                docs = []
                ids = []
                for i, doc in enumerate(self._docs):
                    docs.append(doc)
                    ids.append(f"doc_{i}")
                embeddings = self._encoder.encode(docs, show_progress_bar=False)
                self._collection.add(
                    documents=docs,
                    embeddings=embeddings.tolist(),
                    ids=ids,
                )
        except ImportError:
            pass  # ChromaDB optional for minimal runs

    def query(self, query: str, disaster_type: str = "general",
              top_k: int = 3) -> List[str]:
        """Retrieve relevant knowledge. Tries ChromaDB semantic search first,
        falls back to keyword matching."""

        if self._chroma and hasattr(self, '_collection') and self._collection.count() > 0:
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, disaster_type, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[str]:
        q_embedding = self._encoder.encode([query], show_progress_bar=False)
        results = self._collection.query(
            query_embeddings=q_embedding.tolist(),
            n_results=top_k,
        )
        return results["documents"][0]

    def _keyword_search(self, query: str, disaster_type: str,
                        top_k: int) -> List[str]:
        """Simple keyword overlap scoring for fallback."""
        query_words = set(query)

        # Gather candidates: from specific type + general
        candidates = []
        for dtype in [disaster_type, "general"]:
            for idx in self._doc_index.get(dtype, []):
                candidates.append(self._docs[idx])

        if not candidates:
            return CORE_KNOWLEDGE["general"][:top_k]

        # Score by word overlap
        scored = []
        for doc in candidates:
            doc_words = set(doc)
            score = len(query_words & doc_words)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
