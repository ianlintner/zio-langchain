# ZIO LangChain Evaluation Module

The evaluation module provides a comprehensive framework for assessing LLM and RAG performance in ZIO LangChain applications. It supports both automated metrics and LLM-based evaluation approaches.

## Features

- **Flexible Evaluation Framework**: Core interfaces that support custom evaluation criteria
- **LLM-Based Evaluation**: Uses LLMs to assess outputs based on various quality dimensions
- **Automated Metrics**: Algorithmic evaluation using standard NLP metrics (BLEU, ROUGE, etc.)
- **RAG-Specific Metrics**: Specialized metrics for evaluating RAG system performance
- **Comparative Evaluation**: Tools for comparing outputs from different systems or configurations

## Core Components

### Evaluator

The base trait for all evaluators, providing a unified interface for evaluation:

```scala
trait Evaluator[-I, -O, +R]:
  def evaluate(input: I, output: O): ZIO[Any, EvaluationError, R]
```

### EvaluationResult

Represents the result of an evaluation, including scores and feedback:

```scala
case class EvaluationResult(
  score: Double,
  metrics: Map[String, Double] = Map.empty,
  feedback: Option[String] = None
)
```

## Specialized Evaluators

### RAGEvaluator

Evaluates RAG (Retrieval-Augmented Generation) systems by assessing:
- Relevance of retrieved documents to the query
- Groundedness of the generated response in the documents
- Overall answer quality

```scala
val ragEvaluator = RAGEvaluator.make(llm)
ragEvaluator.evaluate(query, (retrievedDocs, generatedAnswer))
```

### RAGMetricsEvaluator

Provides detailed metrics for RAG systems:
- Context relevance
- Answer relevance
- Faithfulness
- Comprehensiveness
- Groundedness

```scala
val metricsEvaluator = RAGMetricsEvaluator.make(llm)
metricsEvaluator.evaluate(query, retrievedDocs, generatedAnswer, groundTruth)
```

### ChainEvaluator

Evaluates outputs from chains based on custom criteria:

```scala
val chainEvaluator = ChainEvaluator.make[String, String](
  llm,
  Seq("accuracy", "clarity", "completeness"),
  (input, output) => s"Input: $input\n\nOutput: $output"
)
chainEvaluator.evaluate(input, output)
```

### ComparativeEvaluator

Compares outputs from different systems or configurations:

```scala
val comparativeEvaluator = ComparativeEvaluator.make[String, String](
  llm,
  Seq("accuracy", "clarity", "completeness")
)
comparativeEvaluator.evaluate(input, outputs)
```

### AutomatedMetricsEvaluator

Evaluates text using algorithmic metrics without requiring an LLM:

```scala
val automatedEvaluator = AutomatedMetricsEvaluator.make()
automatedEvaluator.evaluate(reference, candidate)
```

Available metrics include:
- Exact Match
- F1 Score
- BLEU
- ROUGE
- Cosine Similarity
- Jaccard Similarity
- Levenshtein Distance

## Usage Example

```scala
import zio.*
import zio.langchain.core.domain.*
import zio.langchain.core.model.LLM
import zio.langchain.evaluation.*
import zio.langchain.integrations.openai.OpenAILLM

// Create an LLM for evaluation
val llm = OpenAILLM.default

// Create an evaluator
val evaluator = ChainEvaluator.make[String, String](
  llm,
  Seq("accuracy", "clarity", "completeness"),
  (input, output) => s"Input: $input\n\nOutput: $output"
)

// Evaluate an output
val program = for
  result <- evaluator.evaluate(
    "Explain quantum computing",
    "Quantum computing uses quantum bits or qubits..."
  )
  _ <- Console.printLine(s"Score: ${result.score}")
  _ <- ZIO.foreach(result.metrics) { case (metric, score) =>
    Console.printLine(s"$metric: $score")
  }
yield ()
```

See the `EvaluationExample.scala` file in the examples module for more detailed usage examples.

## Integration with Other Modules

The evaluation module is designed to work seamlessly with other ZIO LangChain components:

- **LLM Integration**: Uses LLMs for subjective quality assessment
- **RAG Systems**: Specialized evaluators for RAG components
- **Chains**: Evaluates chain outputs with customizable criteria
- **Agents**: Can be used to assess agent responses and reasoning

## Extending the Framework

You can extend the evaluation framework by:

1. Implementing the `Evaluator` trait for custom evaluation logic
2. Creating specialized evaluators for specific use cases
3. Defining custom metrics and evaluation criteria