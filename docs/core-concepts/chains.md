---
title: Chains
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Chains

This document explains the Chain abstraction in ZIO LangChain - how to compose complex workflows from simpler components.

## Table of Contents

- [Introduction to Chains](#introduction-to-chains)
- [Core Chain Interface](#core-chain-interface)
- [Chain Composition](#chain-composition)
- [Common Chain Types](#common-chain-types)
- [Usage Examples](#usage-examples)
- [Custom Chains](#custom-chains)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Introduction to Chains

Chains are a fundamental concept in ZIO LangChain that allow you to compose multiple operations into a unified workflow. They enable you to:

- Connect multiple components into a single processing pipeline
- Reuse and compose modular components
- Build complex LLM applications from simple building blocks
- Transform inputs and outputs between processing steps

## Core Chain Interface

The foundation of chains in ZIO LangChain is the `Chain` trait, which is defined as:

```scala
trait Chain[-R, +E, -I, +O]:
  def run(input: I): ZIO[R, E, O]
  
  def andThen[R1 <: R, E1 >: E, O2](next: Chain[R1, E1, O, O2]): Chain[R1, E1, I, O2] =
    Chain.sequence(this, next)
```

This interface is:

- **Generic**: Works with any input and output types
- **ZIO-based**: Returns ZIO effects for all operations
- **Composable**: Chains can be combined using `andThen`
- **Resource-aware**: Properly handles resource dependencies
- **Error-typed**: Propagates typed errors through the chain

## Chain Composition

The power of chains comes from their composability. You can combine chains in various ways:

### Sequential Composition (andThen)

```scala
val chain1: Chain[Any, Throwable, String, Int] = ???
val chain2: Chain[Any, Throwable, Int, Boolean] = ???

val combinedChain: Chain[Any, Throwable, String, Boolean] = chain1.andThen(chain2)
```

This creates a new chain that passes the output of `chain1` as input to `chain2`.

### Parallel Composition

You can also run chains in parallel by implementing custom combinators:

```scala
def zipWith[R, E, I, O1, O2, O3](
  chain1: Chain[R, E, I, O1],
  chain2: Chain[R, E, I, O2]
)(
  combiner: (O1, O2) => O3
): Chain[R, E, I, O3] = new Chain[R, E, I, O3] {
  override def run(input: I): ZIO[R, E, O3] =
    for
      result1 <- chain1.run(input)
      result2 <- chain2.run(input)
    yield combiner(result1, result2)
}
```

## Common Chain Types

ZIO LangChain provides several built-in chain implementations:

### LLMChain

The most common type of chain, which processes inputs through an LLM:

```scala
class LLMChain[R](llm: LLM, promptTemplate: PromptTemplate) 
  extends Chain[R, LLMError, Map[String, String], String]:
  
  override def run(input: Map[String, String]): ZIO[R, LLMError, String] =
    for
      prompt <- ZIO.attempt(promptTemplate.format(input))
                   .mapError(e => LLMError(e))
      response <- llm.complete(prompt)
    yield response
```

### RetrievalChain

Combines document retrieval with LLM processing:

```scala
class RetrievalChain[R](
  retriever: Retriever, 
  llm: LLM, 
  promptTemplate: RetrievalPromptTemplate
) extends Chain[R, LangChainError, String, String]:
  
  override def run(query: String): ZIO[R, LangChainError, String] =
    for
      documents <- retriever.retrieve(query)
      prompt = promptTemplate.format(query, documents)
      response <- llm.complete(prompt)
    yield response
```

### MapChain

Transforms data between chain steps:

```scala
class MapChain[R, E, I, O1, O2](
  chain: Chain[R, E, I, O1],
  mapper: O1 => O2
) extends Chain[R, E, I, O2]:
  
  override def run(input: I): ZIO[R, E, O2] =
    chain.run(input).map(mapper)
```

## Usage Examples

### Creating a Simple Chain

```scala
import zio.*
import zio.langchain.core.chain.*
import zio.langchain.core.model.LLM
import zio.langchain.core.prompt.PromptTemplate

// Create a prompt template
val promptTemplate = PromptTemplate(
  template = "Answer the following question: {question}",
  inputVariables = Set("question")
)

// Create an LLM chain
val llmChain = new LLMChain(llm, promptTemplate)

// Use the chain
val result = llmChain.run(Map("question" -> "What is ZIO?"))
```

### Building a QA Chain with Retrieval

```scala
import zio.langchain.core.retriever.Retriever
import zio.langchain.core.prompt.RetrievalPromptTemplate

// Define the retrieval template
val retrievalTemplate = RetrievalPromptTemplate(
  template = """Answer the question based on these documents:
               |
               |{context}
               |
               |Question: {question}
               |Answer:""".stripMargin,
  documentVariable = "context"
)

// Create the retrieval chain
val retrievalChain = new RetrievalChain(
  retriever = retriever,
  llm = llm,
  promptTemplate = retrievalTemplate
)

// Use the chain
val answer = retrievalChain.run("What is functional programming?")
```

### Composing Multiple Chains

```scala
// Chain 1: Retrieve relevant documents
val retrievalChain = Chain[Any, Throwable, String, Seq[Document]] { query =>
  retriever.retrieve(query).mapError(e => e.cause)
}

// Chain 2: Format a prompt with retrieved documents
val promptChain = Chain[Any, Throwable, (String, Seq[Document]), String] { 
  case (query, documents) =>
    val context = documents.map(_.content).mkString("\n\n")
    ZIO.succeed(s"Context:\n$context\n\nQuestion: $query\n\nAnswer:")
}

// Chain 3: Process the prompt with an LLM
val llmChain = Chain[Any, Throwable, String, String] { prompt =>
  llm.complete(prompt).mapError(e => e)
}

// Compose the chains together
val qaChain = for
  docs <- retrievalChain
  prompt <- promptChain.contramap[(String, Seq[Document])](query => (query, docs))
  answer <- llmChain
yield answer

// Use the composed chain
val result = qaChain.run("What are the key features of ZIO?")
```

## Custom Chains

You can create custom chains by implementing the `Chain` trait:

```scala
import zio.langchain.core.chain.Chain
import zio.langchain.core.domain.Document
import zio.json.*

// A chain that parses JSON into a Document
class JsonParserChain extends Chain[Any, Throwable, String, Document]:
  override def run(input: String): ZIO[Any, Throwable, Document] =
    ZIO.fromEither(input.fromJson[Document])
      .mapError(error => new RuntimeException(s"JSON parsing error: $error"))
```

## Error Handling

Chains propagate errors through the ZIO effect system. You can handle errors at any point in the chain:

```scala
val robustChain = originalChain.mapError { error =>
  // Transform or log the error
  println(s"Error in chain: $error")
  error
}

// Or use ZIO's error handling operators
val fallbackChain = originalChain.run(input).catchAll { error =>
  // Provide a fallback response
  ZIO.succeed("I couldn't process your request. Please try again.")
}
```

## Best Practices

1. **Composability**: Design chains to be easily composed with others.

2. **Single Responsibility**: Each chain should handle one specific task.

3. **Type Safety**: Leverage Scala's type system to ensure compatible inputs and outputs.

4. **Error Handling**: Provide clear error types and appropriate error handling.

5. **Resource Management**: Use ZIO's dependency management for resource-intensive chains.

6. **Testing**: Write unit tests for individual chains and integration tests for composed chains.

7. **Documentation**: Document the input requirements and output guarantees for each chain.

8. **Reusability**: Avoid hardcoding values that could be parameterized.

9. **Functional Design**:
   ```scala
   // Prefer this:
   def createChain(parameter: String): Chain[R, E, I, O] = ...
   
   // Over this:
   class ParameterizedChain(parameter: String) extends Chain[R, E, I, O] 
   ```

10. **Logging**: Add appropriate logging for debugging complex chain compositions.