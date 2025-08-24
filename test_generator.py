import uuid
from typing import Dict, List
from semantic_chunker import SemanticChunker
import time


class TestGenerator:
    def __init__(self, ai_service, batch_size: int = 5, retry_delay: int = 15):
        self.ai_service = ai_service
        self.chunker = SemanticChunker()
        self.batch_size = batch_size
        self.retry_delay = retry_delay

    def _batch_call(self, prompts: List[str]) -> List[List[str]]:
        """Call AI service in batches with retry on 429 errors. 
        Expect JSON list of strings as output from the model."""
        responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            try:
                batch_responses = self.ai_service.get_answer(batch, [])
                # Always wrap in list of lists
                if isinstance(batch_responses, list):
                    responses.extend(batch_responses)
                else:
                    responses.append(batch_responses)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    print(f"Quota exceeded, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    batch_responses = self.ai_service.get_answer(batch, [])
                    responses.extend(batch_responses if isinstance(batch_responses, list) else [batch_responses])
                else:
                    responses.extend([["ERROR: " + str(e)]] * len(batch))
        return responses

    def generate_mock_questions(self, chunks: List[str], n: int = 3) -> List[List[str]]:
        prompts = [
            f"""
            You are a dataset generator. 
            Based ONLY on the text below, generate exactly {n} diverse and natural user questions.

            ⚠️ IMPORTANT:
            - Return ONLY a valid JSON array of strings.
            - Do NOT add explanations, formatting, or extra text.
            - Example output: ["Question 1?", "Question 2?", "Question 3?"]

            Text:
            {chunk}
            """
            for chunk in chunks
        ]
        return self._batch_call(prompts)

    def paraphrase_questions(self, questions: List[str], n: int = 2) -> List[List[str]]:
        prompts = [
            f"""
            Paraphrase the following question into exactly {n} variations.

            ⚠️ IMPORTANT:
            - Return ONLY a valid JSON array of strings.
            - Do NOT add explanations or other text.
            - Example output: ["Paraphrase 1", "Paraphrase 2"]

            Question:
            {q}
            """
            for q in questions
        ]
        return self._batch_call(prompts)

    def map_questions_to_chunks(self, text: str) -> Dict[str, Dict]:
        chunks = self.chunker.chunk(text)

        # Debug word counts
        total_words = len(text.split())
        chunk_word_counts = [len(c.split()) for c in chunks]
        print(f"📄 Total words in document: {total_words}")
        print(f"✂️ Number of chunks: {len(chunks)}")
        for i, count in enumerate(chunk_word_counts, 1):
            print(f"   - Chunk {i}: {count} words")
        print(f"🔍 Sum of words in chunks: {sum(chunk_word_counts)}")

        q_map = {}

        # Step 1: Generate mock questions
        mock_questions_per_chunk = self.generate_mock_questions(chunks)

        # Step 2: Flatten for paraphrasing
        all_mock_questions = [q for qs in mock_questions_per_chunk for q in qs]
        paraphrased_all = self.paraphrase_questions(all_mock_questions)

        # Step 3: Map back to chunks
        p_idx = 0
        for chunk, mock_qs in zip(chunks, mock_questions_per_chunk):
            qid = str(uuid.uuid4())[:8]
            alt_qs = []
            for _ in mock_qs:
                if p_idx < len(paraphrased_all):
                    alt_qs.extend(paraphrased_all[p_idx])
                p_idx += 1
            q_map[qid] = {
                "chunk": chunk,
                "questions": mock_qs,
                "alt_questions": alt_qs
            }

        return q_map
