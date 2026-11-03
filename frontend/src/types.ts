export interface Review {
  id: number;
  text: string;
  rating?: number | null;
  true_label?: "fake" | "real";
}

export interface ReviewScore {
  id: number;
  fake_prob: number;
  positive_prob: number;
  aspects: string[];
}

export interface Quote {
  review_id: number;
  sentence: string;
  positive_prob: number;
}

export interface AspectStat {
  aspect: string;
  mentions: number;
  positive: number;
  negative: number;
  score: number;
  best: Quote | null;
  worst: Quote | null;
}

export interface Analysis {
  reviews: ReviewScore[];
  aspects: AspectStat[];
  summary: {
    total: number;
    suspected_fake: number;
    fake_share: number;
    avg_rating_all: number | null;
    avg_rating_genuine: number | null;
    detector: string;
    detector_accuracy_on_sample?: number;
  };
}

export interface Citation {
  id: number;
  text: string;
  rating?: number | null;
  fake_prob?: number | null;
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  retrieved: number[];
  excluded_fake: number;
  mode: "llm" | "offline" | "none";
  notice?: string;
}

export interface Scores {
  accuracy: number;
  f1: number;
}

export interface ModelMetrics {
  detector: string;
  transformer?: {
    model: string;
    params_millions: number;
    test_size: number;
    test_onnx_int8: Scores;
    onnx_int8_mb: number;
  };
  classic?: {
    sentiment: { model: string; accuracy: number; f1_negative: number; test_size: number };
    fake_baseline: { model: string; accuracy: number; f1: number; test_size: number };
  };
}

export const FAKE_THRESHOLD = 0.5;
