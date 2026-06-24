"""RAG event memory — semantic search over audio/video/system events.

Self-contained module. Stores events in LanceDB with ONNX embeddings,
supports natural-language queries over historical events.

Usage::

    from memory.indexer import index_event
    from memory.query import query_events

    index_event({"module": "vision", "timestamp": ..., "summary": {"person": 2}})
    results = query_events("有人经过吗？")
"""
