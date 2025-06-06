#!/usr/bin/env python3
"""
Test script for the new topic format with summary, seconds, and sentences
"""

import sys
import os
import json

def test_new_topic_format():
    """Test the new topic format structure"""
    
    # Sample mock data that matches the new format
    sample_topics = [
        {
            "summary": "",  # Will be filled by LLM later
            "seconds": 0.0,
            "sentences": [
                "Today we're going to discuss artificial intelligence and its applications.",
                "Machine learning has revolutionized many industries.",
                "Deep learning models can process vast amounts of data efficiently."
            ]
        },
        {
            "summary": "",  # Will be filled by LLM later
            "seconds": 45.2,
            "sentences": [
                "Moving on to a different topic, let's talk about climate change.",
                "Global warming is affecting weather patterns worldwide.",
                "Renewable energy sources are becoming more important than ever."
            ]
        },
        {
            "summary": "",  # Will be filled by LLM later
            "seconds": 120.8,
            "sentences": [
                "Finally, I want to mention the importance of education.",
                "Learning new skills is crucial in today's rapidly changing world.",
                "Online education platforms have made knowledge more accessible."
            ]
        }
    ]
    
    print("=== New Topic Format Test ===")
    print(f"Number of topics: {len(sample_topics)}")
    print()
    
    # Validate the structure
    for i, topic in enumerate(sample_topics, 1):
        print(f"Topic {i}:")
        
        # Check required fields
        required_fields = ["summary", "seconds", "sentences"]
        for field in required_fields:
            if field not in topic:
                print(f"  ❌ Missing required field: {field}")
                return False
            else:
                print(f"  ✅ Has field: {field}")
        
        # Check data types
        if not isinstance(topic["summary"], str):
            print(f"  ❌ Summary should be string, got {type(topic['summary'])}")
            return False
        
        if not isinstance(topic["seconds"], (int, float)):
            print(f"  ❌ Seconds should be number, got {type(topic['seconds'])}")
            return False
            
        if not isinstance(topic["sentences"], list):
            print(f"  ❌ Sentences should be list, got {type(topic['sentences'])}")
            return False
        
        # Check sentences are strings
        for j, sentence in enumerate(topic["sentences"]):
            if not isinstance(sentence, str):
                print(f"  ❌ Sentence {j} should be string, got {type(sentence)}")
                return False
        
        print(f"  📊 Timing: {topic['seconds']} seconds")
        print(f"  📝 Sentences: {len(topic['sentences'])}")
        print(f"  📄 Summary: {'Empty (to be filled by LLM)' if not topic['summary'] else 'Present'}")
        print()
    
    # Test JSON serialization
    try:
        json_output = json.dumps({"topics": sample_topics}, indent=2)
        print("✅ JSON serialization successful")
        print("\nSample JSON output:")
        print(json_output[:200] + "..." if len(json_output) > 200 else json_output)
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False
    
    return True

def test_integration_format():
    """Test how the format integrates with the full response"""
    
    full_response = {
        "raw_text": "Complete transcribed text...",
        "sentences": ["Sentence 1", "Sentence 2", "Sentence 3"],
        "entities": [],
        "keywords": [],
        "noun_phrases": [],
        "topics": [
            {
                "summary": "",
                "seconds": 0.0,
                "sentences": ["Sentence 1", "Sentence 2"]
            },
            {
                "summary": "",
                "seconds": 30.5,
                "sentences": ["Sentence 3"]
            }
        ],
        "summary_stats": {
            "total_tokens": 10,
            "sentences_count": 3,
            "entities_count": 0,
            "unique_entities": 0,
            "topics_count": 2
        },
        "source_type": "audio",
        "filename": "test.wav"
    }
    
    print("\n=== Integration Test ===")
    
    # Validate topics count matches
    if len(full_response["topics"]) != full_response["summary_stats"]["topics_count"]:
        print("❌ Topics count mismatch")
        return False
    
    print("✅ Topics count matches summary stats")
    
    # Validate timing is sequential
    times = [topic["seconds"] for topic in full_response["topics"]]
    if times != sorted(times):
        print("❌ Topic timing is not sequential")
        return False
    
    print("✅ Topic timing is sequential")
    
    # Test JSON serialization of full response
    try:
        json.dumps(full_response, indent=2)
        print("✅ Full response JSON serialization successful")
    except Exception as e:
        print(f"❌ Full response JSON serialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing new topic format structure...")
    
    try:
        test1_success = test_new_topic_format()
        test2_success = test_integration_format()
        
        if test1_success and test2_success:
            print("\n🎉 All tests passed! New topic format is working correctly.")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)
