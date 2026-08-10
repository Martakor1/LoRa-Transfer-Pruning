from typing import Dict, List, Generator, Optional
from datasets import load_dataset

class MMLUBatchGenerator:
    """
    Generator that yields batches of 64 questions for each MMLU subject
    """

    def __init__(self,
                 subjects: Optional[List[str]] = None,
                 split: str = "test",
                 batch_size: int = 64,
                 shuffle: bool = False,
                 include_metadata: bool = True):
        """
        Initialize the MMLU batch generator

        Args:
            subjects: List of subject names (None = all 57 subjects)
            split: "test", "validation", or "dev"
            batch_size: Number of questions per batch (default: 64)
            shuffle: Whether to shuffle questions within each subject
            include_metadata: Whether to include subject name in output
        """
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_metadata = include_metadata

        # Load all subjects if none specified
        if subjects is None:
            # Get all available subjects
            self.subjects = self._get_all_subjects()
        else:
            self.subjects = subjects

        print(
            f"Initialized MMLU batch generator with {len(self.subjects)} subjects")
        print(f"Split: {split}, Batch size: {batch_size}")

    def _get_all_subjects(self) -> List[str]:
        """Get list of all 57 MMLU subjects"""
        # This is a comprehensive list of all MMLU subjects
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"
        ]

    def _format_question(self, example: Dict, subject: str) -> Dict:
        """Format a single question with its choices"""
        question = example['question']
        choices = example['choices']
        answer = example['answer']

        # Format choices as A, B, C, D
        formatted_choices = "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

        if self.include_metadata:
            return {
                'subject': subject,
                'question': question,
                'choices': choices,
                'formatted_question': f"{question}\n{formatted_choices}",
                'answer': answer,
                'answer_letter': chr(65 + answer) if answer < 4 else '?'
            }
        else:
            return question

    def __iter__(self) -> Generator:
        """
        Generator that yields batches of 64 questions for each subject
        """
        total_subjects = len(self.subjects)

        for subject_idx, subject in enumerate(self.subjects, 1):
            try:
                print(
                    f"\n📚 Loading subject {subject_idx}/{total_subjects}: {subject}")

                # Load the subject
                dataset = load_dataset(
                    "cais/mmlu", subject, trust_remote_code=True)
                split_data = dataset[self.split]

                # Get first 64 questions (or all if less than 64)
                num_questions = min(len(split_data), self.batch_size)
                indices = list(range(num_questions))

                if self.shuffle:
                    import random
                    random.shuffle(indices)

                # Create batch
                batch = []
                for idx in indices:
                    example = split_data[idx]
                    formatted = self._format_question(example, subject)
                    batch.append(formatted)

                print(f"  ✓ Created batch with {len(batch)} questions")

                # Yield the batch with metadata
                if self.include_metadata:
                    yield {
                        'subject': subject,
                        'batch_size': len(batch),
                        'total_questions': len(split_data),
                        'questions': batch
                    }
                else:
                    yield batch

            except Exception as e:
                print(f"  ❌ Error loading {subject}: {e}")
                continue

    def __len__(self) -> int:
        """Return number of subjects (number of batches)"""
        return len(self.subjects)

    def get_subject_batch(self, subject: str) -> Optional[Dict]:
        """Get a single batch for a specific subject"""
        for batch in self:
            if batch['subject'] == subject:
                return batch
        return None
