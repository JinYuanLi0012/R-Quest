// Scores transcribed programmatically from the supplied paper's evaluation tables.
const BENCHMARKS = {
  "qwen": {
    "name": "Qwen3-4B-Base",
    "domains": {
      "math": {
        "headers": [
          "Average",
          "AMC",
          "Minerva",
          "MATH-500",
          "GSM8K",
          "OlympiadBench",
          "AIME 2025",
          "AIME 2024"
        ],
        "rows": [
          [
            "Base model",
            46.17,
            47.34,
            51.84,
            74.2,
            88.63,
            39.11,
            11.35,
            10.73
          ],
          [
            "Base + validity init.",
            47.07,
            49.14,
            53.68,
            74.4,
            89.76,
            41.04,
            10.73,
            10.73
          ],
          [
            "R-Zero",
            49.1,
            54.84,
            54.78,
            76.2,
            92.42,
            42.37,
            12.29,
            10.83
          ],
          [
            "OCNR",
            49.57,
            55.94,
            58.09,
            77.6,
            92.12,
            43.11,
            7.92,
            12.19
          ],
          [
            "R-Diverse",
            49.82,
            55.31,
            57.35,
            77.2,
            92.19,
            43.7,
            9.48,
            13.54
          ],
          [
            "R-Quest",
            51.64,
            57.89,
            58.82,
            77.6,
            92.57,
            46.67,
            13.44,
            14.48
          ]
        ]
      },
      "code": {
        "headers": [
          "Average",
          "HumanEval+",
          "MBPP+"
        ],
        "rows": [
          [
            "Base model",
            58.79,
            54.88,
            62.7
          ],
          [
            "Base + validity init.",
            59.63,
            55.49,
            63.76
          ],
          [
            "R-Zero",
            59.27,
            56.1,
            62.43
          ],
          [
            "OCNR",
            60.28,
            57.32,
            63.23
          ],
          [
            "R-Diverse",
            60.5,
            56.71,
            64.29
          ],
          [
            "R-Quest",
            62.69,
            59.76,
            65.61
          ]
        ]
      },
      "general": {
        "headers": [
          "Average",
          "SuperGPQA",
          "MMLU-Pro",
          "BBEH"
        ],
        "rows": [
          [
            "Base model",
            28.79,
            26.21,
            51.41,
            8.74
          ],
          [
            "Base + validity init.",
            29.4,
            26.28,
            51.58,
            10.35
          ],
          [
            "R-Zero",
            31.21,
            28.66,
            54.01,
            10.95
          ],
          [
            "OCNR",
            31.56,
            29.05,
            54.82,
            10.82
          ],
          [
            "R-Diverse",
            31.76,
            28.98,
            55.79,
            10.51
          ],
          [
            "R-Quest",
            32.75,
            29.4,
            57.36,
            11.5
          ]
        ]
      }
    }
  },
  "octo": {
    "name": "OctoThinker-3B",
    "domains": {
      "math": {
        "headers": [
          "Average",
          "AMC",
          "Minerva",
          "MATH-500",
          "GSM8K",
          "OlympiadBench",
          "AIME 2025",
          "AIME 2024"
        ],
        "rows": [
          [
            "Base model",
            24.9,
            25,
            23.16,
            41.8,
            69.14,
            11.85,
            0,
            3.33
          ],
          [
            "Base + validity init.",
            25.22,
            19.84,
            26.84,
            43.4,
            69.9,
            13.63,
            0,
            2.92
          ],
          [
            "R-Zero",
            28.25,
            25.47,
            29.78,
            48.8,
            75.89,
            15.11,
            0.1,
            2.6
          ],
          [
            "OCNR",
            28.12,
            30,
            28.68,
            49.6,
            73.16,
            15.41,
            0,
            0
          ],
          [
            "R-Diverse",
            28.61,
            29.84,
            27.21,
            49.2,
            74.53,
            16.15,
            0,
            3.33
          ],
          [
            "R-Quest",
            30.69,
            32.27,
            29.41,
            55.8,
            76.42,
            17.19,
            0.31,
            3.44
          ]
        ]
      },
      "code": {
        "headers": [
          "Average",
          "HumanEval+",
          "MBPP+"
        ],
        "rows": [
          [
            "Base model",
            13.67,
            14.63,
            12.7
          ],
          [
            "Base + validity init.",
            14.77,
            15.24,
            14.29
          ],
          [
            "R-Zero",
            14.14,
            15.85,
            12.43
          ],
          [
            "OCNR",
            15.6,
            15.85,
            15.34
          ],
          [
            "R-Diverse",
            14.98,
            16.46,
            13.49
          ],
          [
            "R-Quest",
            18.9,
            17.68,
            20.11
          ]
        ]
      },
      "general": {
        "headers": [
          "Average",
          "SuperGPQA",
          "MMLU-Pro",
          "BBEH"
        ],
        "rows": [
          [
            "Base model",
            8.3,
            10.78,
            12.14,
            1.97
          ],
          [
            "Base + validity init.",
            9.04,
            11.42,
            13.53,
            2.17
          ],
          [
            "R-Zero",
            14.1,
            13.98,
            22.25,
            6.08
          ],
          [
            "OCNR",
            13.83,
            13.89,
            21.86,
            5.73
          ],
          [
            "R-Diverse",
            14.5,
            14.56,
            22.91,
            6.04
          ],
          [
            "R-Quest",
            15.5,
            14.96,
            25,
            6.55
          ]
        ]
      }
    }
  }
};
