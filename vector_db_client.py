import json
from typing import List, Dict, Any

class VectorDBClient:
    """Example client for vector database integration"""

    def prepare_for_indexing(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare structured data for vector database indexing"""
        documents = []

        # Index full text
        documents.append({
            "id": f"{structured_data['filename']}_full",
            "text": structured_data["raw_text"],
            "metadata": {
                "source": structured_data["filename"],
                "type": structured_data["source_type"],
                "content_type": "full_transcript"
            }
        })

        # Index sentences separately
        for i, sentence in enumerate(structured_data["sentences"]):
            documents.append({
                "id": f"{structured_data['filename']}_sent_{i}",
                "text": sentence,
                "metadata": {
                    "source": structured_data["filename"],
                    "type": structured_data["source_type"],
                    "content_type": "sentence",
                    "sentence_index": i
                }
            })

        # Index entities
        for entity in structured_data["entities"]:
            documents.append({
                "id": f"{structured_data['filename']}_entity_{entity['start']}",
                "text": entity["text"],
                "metadata": {
                    "source": structured_data["filename"],
                    "type": structured_data["source_type"],
                    "content_type": "entity",
                    "entity_label": entity["label"],
                    "entity_description": entity["description"]
                }
            })

        return documents
