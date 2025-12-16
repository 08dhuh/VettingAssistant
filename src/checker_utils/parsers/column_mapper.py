import json
from typing import Optional, List, Dict
import pandas as pd
import openai


class ColumnMapper:
    """
    Uses LLM to intelligently map DataFrame columns to semantic roles.
    Avoids fragile keyword matching that leads to false positives.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the column mapper.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (using mini for speed/cost)
        """
        self.api_key = api_key
        self.model = model

        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None

    def map_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        context: str = ""
    ) -> Dict[str, Optional[str]]:
        """
        Map DataFrame columns to required semantic roles using LLM.

        Args:
            df: DataFrame to analyze
            required_columns: List of semantic column types needed
                            (e.g., ["mean", "observations", "highest_value", "second_highest"])
            context: Additional context to help with mapping

        Returns:
            Dictionary mapping semantic role to actual column name
            Example: {"mean": "income", "observations": "number of observations"}
        """
        if not self.client:
            # Fallback to simple keyword matching if no API key
            return self._fallback_mapping(df, required_columns)

        # Prepare column information
        column_info = []
        for col in df.columns:
            sample_values = df[col].dropna().head(3).tolist()
            column_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "sample_values": [str(v) for v in sample_values]
            })

        prompt = f"""
You are a data analyst. Map DataFrame columns to their semantic roles.

Available columns:
{json.dumps(column_info, indent=2)}

Required semantic roles:
{json.dumps(required_columns)}

{f"Context: {context}" if context else ""}

IMPORTANT GUIDELINES:
- "mean" or "average": The column containing calculated mean/average values
- "observations" or "sample_size": The column containing COUNT of observations (not values, not amounts)
- "highest_value" or "first_highest": The column with the single highest individual value
- "second_highest": The column with the second highest individual value
- "year" or "period": Time period column

Look at:
1. Column names (exact matches are best)
2. Data types (counts are usually integers, means can be floats)
3. Sample values (observations should be reasonable counts like 1988, 2099365, not dollar amounts)

Return ONLY valid JSON mapping semantic roles to actual column names.
Use null if a required column cannot be found.

Format:
{{
  "semantic_role": "actual_column_name",
  "another_role": "another_column_name",
  "not_found_role": null
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst expert at mapping column names to semantic meanings. Return ONLY valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=500,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Clean markdown if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            mapping = json.loads(response_text)

            return mapping

        except Exception as e:
            print(f"[ColumnMapper] LLM mapping failed: {e}. Falling back to keyword matching.")
            return self._fallback_mapping(df, required_columns)

    def _fallback_mapping(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Optional[str]]:
        """
        Fallback to keyword-based mapping if LLM is unavailable.
        """
        mapping = {}

        keyword_map = {
            "mean": ["mean", "average", "avg", "income", "value", "amount"],
            "observations": ["observations", "number of observations", "count", "n", "sample", "size"],
            "sample_size": ["observations", "number of observations", "count", "n", "sample", "size"],
            "highest_value": ["highest", "first", "1st", "max", "top1"],
            "first_highest": ["highest", "first", "1st", "max", "top1"],
            "second_highest": ["second", "2nd", "top2"],
            "year": ["year", "period", "date", "time"],
        }

        for role in required_columns:
            keywords = keyword_map.get(role, [role])
            found_col = self._find_column(df, keywords)
            mapping[role] = found_col

        return mapping

    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find column matching any of the keywords (case-insensitive)"""
        # Exact match first
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for keyword in keywords:
                if col_lower == keyword.lower().strip():
                    return col

        # Partial match second
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in keywords:
                if keyword.lower() in col_lower:
                    return col

        return None
