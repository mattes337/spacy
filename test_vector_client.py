#!/usr/bin/env python3
"""
Test script for the VectorDBClient
"""

import json
from vector_db_client import VectorDBClient

def test_vector_client():
    """Test the vector database client functionality"""
    
    # Sample structured data from audio processing
    test_data = {
        "filename": "sample_audio.wav",
        "source_type": "audio",
        "raw_text": "Hello world, this is a test transcription with some entities like John Doe and New York.",
        "sentences": [
            "Hello world, this is a test transcription with some entities like John Doe and New York."
        ],
        "entities": [
            {
                "text": "John Doe",
                "label": "PERSON",
                "description": "People, including fictional",
                "start": 65,
                "end": 73
            },
            {
                "text": "New York",
                "label": "GPE",
                "description": "Countries, cities, states",
                "start": 78,
                "end": 86
            }
        ],
        "keywords": [
            {"text": "Hello", "lemma": "hello", "pos": "INTJ", "is_stop": False},
            {"text": "world", "lemma": "world", "pos": "NOUN", "is_stop": False},
            {"text": "test", "lemma": "test", "pos": "NOUN", "is_stop": False},
            {"text": "transcription", "lemma": "transcription", "pos": "NOUN", "is_stop": False}
        ],
        "noun_phrases": ["test transcription", "John Doe", "New York"],
        "summary_stats": {
            "total_tokens": 16,
            "sentences_count": 1,
            "entities_count": 2,
            "unique_entities": 2
        }
    }
    
    # Initialize client
    client = VectorDBClient()
    
    # Prepare documents for indexing
    documents = client.prepare_for_indexing(test_data)
    
    # Print results
    print("Vector Database Client Test Results")
    print("=" * 50)
    print(f"Input filename: {test_data['filename']}")
    print(f"Source type: {test_data['source_type']}")
    print(f"Generated {len(documents)} documents for indexing")
    print()
    
    # Show each document type
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"  ID: {doc['id']}")
        print(f"  Content Type: {doc['metadata']['content_type']}")
        print(f"  Text: {doc['text'][:100]}{'...' if len(doc['text']) > 100 else ''}")
        print(f"  Metadata: {json.dumps(doc['metadata'], indent=4)}")
        print()
    
    # Verify document structure
    expected_types = ["full_transcript", "sentence", "entity", "entity"]
    actual_types = [doc['metadata']['content_type'] for doc in documents]
    
    print("Validation:")
    print(f"Expected document types: {expected_types}")
    print(f"Actual document types: {actual_types}")
    print(f"✅ Test passed!" if actual_types == expected_types else "❌ Test failed!")
    
    return documents

if __name__ == "__main__":
    try:
        test_vector_client()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
