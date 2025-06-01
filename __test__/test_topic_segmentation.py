#!/usr/bin/env python3
"""
Test script for topic segmentation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import TextProcessor
from config import config

def test_topic_segmentation():
    """Test topic segmentation with sample text"""
    
    # Sample text with multiple topics
    sample_text = """
    Today we're going to discuss artificial intelligence and its applications. 
    Machine learning has revolutionized many industries. Deep learning models 
    can process vast amounts of data efficiently.
    
    Moving on to a different topic, let's talk about climate change. 
    Global warming is affecting weather patterns worldwide. 
    Renewable energy sources are becoming more important than ever.
    
    Finally, I want to mention the importance of education. 
    Learning new skills is crucial in today's rapidly changing world. 
    Online education platforms have made knowledge more accessible.
    """
    
    print("=== Topic Segmentation Test ===")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Topic segmentation enabled: {config.ENABLE_TOPIC_SEGMENTATION}")
    print(f"Similarity threshold: {config.TOPIC_SIMILARITY_THRESHOLD}")
    print(f"Min topic sentences: {config.MIN_TOPIC_SENTENCES}")
    print()
    
    # Initialize processor
    processor = TextProcessor()
    
    # Test topic segmentation
    try:
        topics = processor.segment_text_by_topic(sample_text)
        
        print(f"Number of topics identified: {len(topics)}")
        print()
        
        for i, topic in enumerate(topics, 1):
            print(f"Topic {i}:")
            print(f"  Length: {len(topic)} characters")
            print(f"  Content: {topic[:100]}...")
            print()
            
        # Test full structured processing
        print("=== Full Structured Processing Test ===")
        structured_data = processor.structure_text_with_spacy(sample_text)
        
        print(f"Topics in structured data: {len(structured_data.get('topics', []))}")
        print(f"Summary stats: {structured_data.get('summary_stats', {})}")
        
        return True
        
    except Exception as e:
        print(f"Error during topic segmentation: {e}")
        return False

def test_edge_cases():
    """Test edge cases for topic segmentation"""
    
    print("\n=== Edge Cases Test ===")
    
    processor = TextProcessor()
    
    # Test with very short text
    short_text = "This is a very short text."
    topics = processor.segment_text_by_topic(short_text)
    print(f"Short text topics: {len(topics)}")
    
    # Test with empty text
    empty_text = ""
    topics = processor.segment_text_by_topic(empty_text)
    print(f"Empty text topics: {len(topics)}")
    
    # Test with single sentence
    single_sentence = "This is just one sentence."
    topics = processor.segment_text_by_topic(single_sentence)
    print(f"Single sentence topics: {len(topics)}")

if __name__ == "__main__":
    print("Testing topic segmentation functionality...")
    
    try:
        success = test_topic_segmentation()
        test_edge_cases()
        
        if success:
            print("\n✅ Topic segmentation tests completed successfully!")
        else:
            print("\n❌ Topic segmentation tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)
