# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KorQuAD v1.0에 대한 공식 평가 스크립트 """

import datasets

from .evaluate import evaluate


_CITATION = """\
@{,
  title = {KorQuAD1.0: Korean QA Dataset for Machine Reading Comprehension},
  author={Seungyoung Lim, Myungji Kim, Jooyoul Lee},
  year = {2019},
}
"""


_DESCRIPTION = """
This metric wrap the official scoring script for version 1 of Korean Question Answering Dataset(KorQuAD).

We present Korean Question Answering Dataset(KorQuAD), a large-scale Korean dataset
for extractive machine reading comprehension task. It consists of 70,000+ human generated question-answer pairs
on Korean Wikipedia articles. We release KorQuAD1.0 and launch a challenge at https://KorQuAD.github.io to encourage
the development of multilingual natural language processing research.
"""

_KWARGS_DESCRIPTION = """
Computes KorQuAD scores (F1 and EM).
Args:
    predictions: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair as given in the references (see below)
        - 'prediction_text': the text of the answer
    references: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair (see above),
        - 'answers': a Dict in the KorQuAD dataset format
            {
                'text': list of possible texts for the answer, as a list of strings
                'answer_start': list of start positions for the answer, as a list of ints
            }
            Note that answer_start values are not taken into account to compute the metric.
Returns:
    'exact_match': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
"""


class Korquad(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "id": datasets.Value("string"),
                        "prediction_text": datasets.Value("string"),
                    },
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            codebase_urls=["https://korquad.github.io/category/1.0_KOR"],
            reference_urls=["https://korquad.github.io/category/1.0_KOR"],
        )

    def _compute(self, predictions, references):
        pred_dict = {
            prediction["id"]: prediction["prediction_text"]
            for prediction in predictions
        }
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [
                                    {"text": answer_text}
                                    for answer_text in ref["answers"]["text"]
                                ],
                                "id": ref["id"],
                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        score = evaluate(dataset=dataset, predictions=pred_dict)
        return score
