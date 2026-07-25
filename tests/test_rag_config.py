from src.rag.config import RagConfig


def test_rag_config_default_matches_documented_values():
    config = RagConfig.default()

    assert config.chunking.target_tokens == 500
    assert config.chunking.overlap_tokens == 75
    assert config.embedding.model == "Qwen/Qwen3-Embedding-0.6B"
    assert config.vector_store.backend == "pgvector"


def test_rag_config_load_reads_configs_rag_yaml():
    config = RagConfig.load()

    assert config.chunking.chunking_version == "recursive-500-v1"
    assert config.embedding.device in {"auto", "cpu", "cuda"}
    assert config.generation.model
    assert config.vector_store.table == "rag_chunks"
