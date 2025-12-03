import re
from typing import List, Dict, Set


class DataPreprocessor:
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    @staticmethod
    def combine_title_abstract(title: str, abstract: str) -> str:
        title = DataPreprocessor.clean_text(title)
        abstract = DataPreprocessor.clean_text(abstract)
        
        if title and not title.endswith('.'):
            return f"{title}. {abstract}"
        else:
            return f"{title} {abstract}"
    
    @staticmethod
    def normalize_keyphrases(keyphrases: List[str]) -> List[str]:
        if not keyphrases:
            return []
        
        seen = set()
        normalized = []
        for kp in keyphrases:
            kp_clean = DataPreprocessor.clean_text(kp)
            if kp_clean and kp_clean.lower() not in seen:
                seen.add(kp_clean.lower())
                normalized.append(kp_clean)
        
        return normalized
    
    @staticmethod
    def filter_by_prmu(item: Dict, prmu_filter: Set[str] = None) -> Dict:
        if prmu_filter is None:
            return item
        
        keyphrases = item.get('keyphrases', [])
        prmu = item.get('prmu', [])
        
        if len(keyphrases) != len(prmu):
            return item
        
        filtered_keyphrases = []
        filtered_prmu = []
        for kp, p in zip(keyphrases, prmu):
            if p in prmu_filter:
                filtered_keyphrases.append(kp)
                filtered_prmu.append(p)
        
        result = item.copy()
        result['keyphrases'] = filtered_keyphrases
        result['prmu'] = filtered_prmu
        return result
    
    @staticmethod
    def prepare_for_training(item: Dict) -> Dict:
        title = item.get('title', '')
        abstract = item.get('abstract', '')
        keyphrases = item.get('keyphrases', [])
        
        text = DataPreprocessor.combine_title_abstract(title, abstract)
        normalized_kps = DataPreprocessor.normalize_keyphrases(keyphrases)
        
        result = {
            'id': item.get('id', ''),
            'text': text,
            'title': DataPreprocessor.clean_text(title),
            'abstract': DataPreprocessor.clean_text(abstract),
            'keyphrases': normalized_kps,
            'prmu': item.get('prmu', [])
        }
        
        return result
