# ZIO LangChain Parsers

This module provides structured output parsing capabilities for ZIO LangChain, enabling type-safe interactions with LLMs by converting text outputs into strongly typed Scala objects.

## Features

- Generic and type-safe output parsing
- JSON schema-based output parsing with validation
- Retry mechanisms for handling parsing failures
- Integration with ZIO JSON for parsing
- Proper error handling with specific parser error types

## Usage

### Basic JSON Parsing

```scala
import zio.*
import zio.json.*
import zio.langchain.parsers.*
import zio.langchain.core.model.LLM

// Define a case class for structured output
case class MovieReview(
  title: String,
  year: Int,
  rating: Double,
  review: String,
  pros: List[String],
  cons: List[String]
)

// JSON encoder/decoder for MovieReview
implicit val movieReviewEncoder: JsonEncoder[MovieReview] = DeriveJsonEncoder.gen[MovieReview]
implicit val movieReviewDecoder: JsonDecoder[MovieReview] = DeriveJsonDecoder.gen[MovieReview]

// Create a JSON parser for MovieReview
val parser = OutputParser.json[MovieReview]()

// Use the parser with an LLM
val result = for
  llm <- ZIO.service[LLM]
  prompt = "Write a detailed review of the movie 'The Matrix' (1999)."
  structuredParser = StructuredOutputParser.forJson[MovieReview]()
  review <- structuredParser.generateStructured(prompt, llm)
yield review
```

### Using LLM Extension Methods

```scala
import zio.*
import zio.json.*
import zio.langchain.parsers.*
import zio.langchain.core.model.LLM

// Define a case class and JSON codec as above

// Use the LLM extension methods
val result = for
  llm <- ZIO.service[LLM]
  prompt = "Write a detailed review of the movie 'Inception' (2010)."
  review <- llm.completeStructured(prompt, OutputParser.json[MovieReview]())
yield review
```

### Chat-Based Structured Output

```scala
import zio.*
import zio.json.*
import zio.langchain.parsers.*
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.ChatMessage

// Define a case class and JSON codec as above

// Use with chat messages
val result = for
  llm <- ZIO.service[LLM]
  messages = Seq(
    ChatMessage.system("You are a helpful movie critic assistant."),
    ChatMessage.user("Can you review the movie 'Interstellar' (2014)?")
  )
  review <- llm.completeChatStructured(messages, OutputParser.json[MovieReview]())
yield review
```

### Using JSON Schema Validation

```scala
import zio.*
import zio.json.*
import zio.langchain.parsers.*
import zio.langchain.core.model.LLM

// Define a case class and JSON codec as above

// Create a JSON schema for MovieReview
val schema = JsonSchema.fromType[MovieReview]

// Create a structured parser with schema validation
val structuredParser = StructuredOutputParser.withJsonSchema[MovieReview](schema)

// Use the parser with an LLM
val result = for
  llm <- ZIO.service[LLM]
  prompt = "Write a detailed review of the movie 'The Shawshank Redemption' (1994)."
  review <- structuredParser.generateStructured(prompt, llm)
yield review
```

### Handling Parsing Failures with Retry

```scala
import zio.*
import zio.json.*
import zio.langchain.parsers.*
import zio.langchain.core.model.LLM

// Define a case class and JSON codec as above

// Create a JSON parser for MovieReview
val parser = OutputParser.json[MovieReview]()

// Use the parser with retry capability
val result = for
  llm <- ZIO.service[LLM]
  text <- llm.complete("Write a detailed review of the movie 'Pulp Fiction' (1994).")
  review <- parser.parseWithRetry(text, llm, maxRetries = 2)
yield review
```

## Custom Parsers

You can create custom parsers for specific formats:

```scala
// Simple parser that applies a function
val simpleParser = OutputParser[Int](
  text => text.trim.toInt,
  "an integer"
)

// Regex-based parser
val regexParser = OutputParser.regex[Tuple2[String, Int]](
  "Name: (.*), Age: (\\d+)",
  groups => (groups(1), groups(2).toInt),
  "text in the format 'Name: John, Age: 30'"
)

// Parser with validation
val validatedParser = OutputParser.withValidation[Int](
  OutputParser[Int](_.trim.toInt, "an integer"),
  n => n > 0,
  n => s"Expected a positive integer, but got $n"
)
```

## Error Handling

The parsers use the `OutputParsingError` type for error handling, which includes:

- The underlying cause of the error
- A descriptive error message
- The original output that failed to parse (optional)

This allows for detailed error reporting and debugging.