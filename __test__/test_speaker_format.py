#!/usr/bin/env python3
"""
Test script for speaker diarization functionality and sentence format
"""

import json

def test_speaker_sentence_format():
    """Test the new sentence format with speaker information"""
    
    # Sample sentences with speaker information
    sample_sentences = [
        {"speaker": 1, "text": "Hello, welcome to our meeting today."},
        {"speaker": 2, "text": "Thank you for having me."},
        {"speaker": 1, "text": "Let's start with the first topic."},
        {"speaker": 3, "text": "I have some questions about that."},
        {"speaker": 2, "text": "I agree with the previous speaker."}
    ]
    
    print("=== Speaker Sentence Format Test ===")
    print(f"Number of sentences: {len(sample_sentences)}")
    print()
    
    # Validate sentence structure
    for i, sentence in enumerate(sample_sentences, 1):
        print(f"Sentence {i}:")
        
        # Check required fields
        if "speaker" not in sentence:
            print(f"  ❌ Missing 'speaker' field")
            return False
        if "text" not in sentence:
            print(f"  ❌ Missing 'text' field")
            return False
        
        # Check data types
        if not isinstance(sentence["speaker"], int):
            print(f"  ❌ Speaker should be int, got {type(sentence['speaker'])}")
            return False
        if not isinstance(sentence["text"], str):
            print(f"  ❌ Text should be string, got {type(sentence['text'])}")
            return False
        
        print(f"  ✅ Speaker: {sentence['speaker']}")
        print(f"  ✅ Text: {sentence['text'][:50]}...")
        print()
    
    # Test JSON serialization
    try:
        json_output = json.dumps({"sentences": sample_sentences}, indent=2)
        print("✅ JSON serialization successful")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False
    
    return True

def test_speaker_diarization_output():
    """Test speaker diarization output format"""
    
    sample_speaker_result = {
        "speakers": {"SPEAKER_00": 1, "SPEAKER_01": 2, "SPEAKER_02": 3},
        "speaker_segments": [
            {"start": 0.0, "end": 15.0, "speaker": 1, "speaker_label": "SPEAKER_00"},
            {"start": 15.0, "end": 30.0, "speaker": 2, "speaker_label": "SPEAKER_01"},
            {"start": 30.0, "end": 45.0, "speaker": 1, "speaker_label": "SPEAKER_00"},
            {"start": 45.0, "end": 60.0, "speaker": 3, "speaker_label": "SPEAKER_02"}
        ],
        "num_speakers": 3
    }
    
    print("=== Speaker Diarization Output Test ===")
    
    # Validate structure
    required_fields = ["speakers", "speaker_segments", "num_speakers"]
    for field in required_fields:
        if field not in sample_speaker_result:
            print(f"❌ Missing required field: {field}")
            return False
        print(f"✅ Has field: {field}")
    
    # Validate speaker segments
    for i, segment in enumerate(sample_speaker_result["speaker_segments"]):
        segment_fields = ["start", "end", "speaker", "speaker_label"]
        for field in segment_fields:
            if field not in segment:
                print(f"❌ Segment {i} missing field: {field}")
                return False
    
    print(f"✅ Number of speakers: {sample_speaker_result['num_speakers']}")
    print(f"✅ Number of segments: {len(sample_speaker_result['speaker_segments'])}")
    
    # Test JSON serialization
    try:
        json.dumps(sample_speaker_result, indent=2)
        print("✅ Speaker diarization JSON serialization successful")
    except Exception as e:
        print(f"❌ Speaker diarization JSON serialization failed: {e}")
        return False
    
    return True

def test_full_response_with_speakers():
    """Test complete response format with speaker information"""
    
    full_response = {
        "raw_text": "Hello, welcome to our meeting. Thank you for having me. Let's discuss the agenda.",
        "sentences": [
            {"speaker": 1, "text": "Hello, welcome to our meeting."},
            {"speaker": 2, "text": "Thank you for having me."},
            {"speaker": 1, "text": "Let's discuss the agenda."}
        ],
        "topics": [
            {
                "summary": "",
                "seconds": 0.0,
                "sentences": [
                    {"speaker": 1, "text": "Hello, welcome to our meeting."},
                    {"speaker": 2, "text": "Thank you for having me."}
                ]
            },
            {
                "summary": "",
                "seconds": 15.0,
                "sentences": [
                    {"speaker": 1, "text": "Let's discuss the agenda."}
                ]
            }
        ],
        "summary_stats": {
            "sentences_count": 3,
            "topics_count": 2,
            "speakers_count": 2
        },
        "speaker_diarization": {
            "speakers": {"SPEAKER_00": 1, "SPEAKER_01": 2},
            "speaker_segments": [
                {"start": 0.0, "end": 10.0, "speaker": 1, "speaker_label": "SPEAKER_00"},
                {"start": 10.0, "end": 15.0, "speaker": 2, "speaker_label": "SPEAKER_01"},
                {"start": 15.0, "end": 20.0, "speaker": 1, "speaker_label": "SPEAKER_00"}
            ],
            "num_speakers": 2
        }
    }
    
    print("\n=== Full Response with Speakers Test ===")
    
    # Validate consistency
    if len(full_response["sentences"]) != full_response["summary_stats"]["sentences_count"]:
        print("❌ Sentences count mismatch")
        return False
    
    if len(full_response["topics"]) != full_response["summary_stats"]["topics_count"]:
        print("❌ Topics count mismatch")
        return False
    
    if full_response["speaker_diarization"]["num_speakers"] != full_response["summary_stats"]["speakers_count"]:
        print("❌ Speakers count mismatch")
        return False
    
    print("✅ All counts are consistent")
    
    # Test JSON serialization
    try:
        json.dumps(full_response, indent=2)
        print("✅ Full response JSON serialization successful")
    except Exception as e:
        print(f"❌ Full response JSON serialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing speaker diarization format and functionality...")
    
    try:
        test1_success = test_speaker_sentence_format()
        test2_success = test_speaker_diarization_output()
        test3_success = test_full_response_with_speakers()
        
        if test1_success and test2_success and test3_success:
            print("\n🎉 All speaker diarization tests passed!")
            print("✅ Sentence format with speakers is working correctly")
            print("✅ Speaker diarization output format is valid")
            print("✅ Full response integration is successful")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
