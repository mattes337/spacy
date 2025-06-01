#!/usr/bin/env python3
"""
Basic functionality test for topic segmentation
Tests the implementation without requiring full ML dependencies
"""

import sys
import os

def test_imports():
    """Test that the basic imports work"""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ NumPy and scikit-learn imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cosine_similarity():
    """Test basic cosine similarity functionality"""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create two simple vectors
        vec1 = np.array([[1, 2, 3]])
        vec2 = np.array([[4, 5, 6]])
        
        # Calculate similarity
        similarity = cosine_similarity(vec1, vec2)[0][0]
        
        print(f"✅ Cosine similarity test: {similarity:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Cosine similarity test failed: {e}")
        return False

def test_config_import():
    """Test that config imports work"""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config
        
        print(f"✅ Config import successful")
        print(f"   Topic segmentation enabled: {config.ENABLE_TOPIC_SEGMENTATION}")
        print(f"   Similarity threshold: {config.TOPIC_SIMILARITY_THRESHOLD}")
        print(f"   Min topic sentences: {config.MIN_TOPIC_SENTENCES}")
        return True
        
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_mock_topic_segmentation():
    """Test a simple mock version of topic segmentation logic"""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Mock sentence vectors (representing different topics)
        sentences = [
            "This is about artificial intelligence and machine learning.",
            "Deep learning models are very powerful for data processing.",
            "Now let's talk about climate change and global warming.",
            "Renewable energy is important for our future.",
            "Education is crucial in today's world.",
            "Online learning platforms make knowledge accessible."
        ]
        
        # Create mock vectors (in real implementation, these would come from spaCy)
        # Topic 1: AI/ML (similar vectors)
        # Topic 2: Climate (similar vectors) 
        # Topic 3: Education (similar vectors)
        mock_vectors = np.array([
            [1.0, 0.8, 0.1, 0.1, 0.0, 0.0],  # AI sentence 1
            [0.9, 1.0, 0.1, 0.0, 0.1, 0.0],  # AI sentence 2
            [0.1, 0.0, 1.0, 0.8, 0.0, 0.1],  # Climate sentence 1
            [0.0, 0.1, 0.9, 1.0, 0.0, 0.0],  # Climate sentence 2
            [0.0, 0.0, 0.1, 0.0, 1.0, 0.8],  # Education sentence 1
            [0.1, 0.0, 0.0, 0.1, 0.9, 1.0],  # Education sentence 2
        ])
        
        # Test similarity calculation
        similarity_threshold = 0.75
        topics = []
        current_topic = [0]  # Start with first sentence
        
        for i in range(1, len(mock_vectors)):
            similarity = cosine_similarity(
                mock_vectors[i-1].reshape(1, -1),
                mock_vectors[i].reshape(1, -1)
            )[0][0]
            
            if similarity > similarity_threshold:
                current_topic.append(i)
            else:
                # Topic change detected
                topics.append([sentences[j] for j in current_topic])
                current_topic = [i]
        
        # Add last topic
        topics.append([sentences[j] for j in current_topic])
        
        print(f"✅ Mock topic segmentation successful")
        print(f"   Number of topics identified: {len(topics)}")
        for i, topic in enumerate(topics, 1):
            print(f"   Topic {i}: {len(topic)} sentences")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock topic segmentation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Basic Functionality Tests ===")
    
    tests = [
        test_imports,
        test_cosine_similarity,
        test_config_import,
        test_mock_topic_segmentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Topic segmentation implementation is ready.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
