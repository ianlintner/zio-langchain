package zio.langchain.examples

import zio.*
import zio.json.*
import zio.Console.*

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.ChatMessage
import zio.langchain.integrations.openai.{OpenAIConfig, OpenAILLM}
import zio.langchain.parsers.{OutputParser, JsonSchemaOutputParser, StructuredOutputParser, JsonSchema}

/**
 * Example demonstrating the use of structured output parsing with LLMs.
 * This example shows how to:
 * 1. Parse LLM outputs into strongly typed Scala objects
 * 2. Use JSON schema-based output parsing
 * 3. Handle parsing failures with retry mechanisms
 */
object StructuredOutputExample extends ZIOAppDefault:

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

  // Example 1: Basic JSON parsing
  def basicJsonParsingExample(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a JSON parser for MovieReview
    val parser = OutputParser.json[MovieReview]()
    
    // Prompt for the LLM
    val prompt = """
      |Write a detailed review of the movie "The Matrix" (1999).
      |Include the title, year, your rating out of 10, a review, and lists of pros and cons.
      |""".stripMargin
    
    // Use the StructuredOutputParser to generate and parse the output
    val structuredParser = StructuredOutputParser.forJson[MovieReview]()
    
    for
      _ <- printLine("Example 1: Basic JSON Parsing")
      _ <- printLine("Generating structured movie review...")
      review <- structuredParser.generateStructured(prompt, llm)
      _ <- printLine(s"Parsed review: ${review.toJson}")
      _ <- printLine(s"Movie: ${review.title} (${review.year})")
      _ <- printLine(s"Rating: ${review.rating}/10")
      _ <- printLine(s"Review: ${review.review}")
      _ <- printLine("Pros:")
      _ <- ZIO.foreach(review.pros)(pro => printLine(s"- $pro"))
      _ <- printLine("Cons:")
      _ <- ZIO.foreach(review.cons)(con => printLine(s"- $con"))
      _ <- printLine("")
    yield ()

  // Example 2: Using the LLM extension methods
  def llmExtensionExample(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a JSON parser for MovieReview
    val parser = OutputParser.json[MovieReview]()
    
    // Prompt for the LLM
    val prompt = """
      |Write a detailed review of the movie "Inception" (2010).
      |Include the title, year, your rating out of 10, a review, and lists of pros and cons.
      |""".stripMargin
    
    for
      _ <- printLine("Example 2: Using LLM Extension Methods")
      _ <- printLine("Generating structured movie review...")
      review <- llm.completeStructured(prompt, parser)
      _ <- printLine(s"Parsed review: ${review.toJson}")
      _ <- printLine(s"Movie: ${review.title} (${review.year})")
      _ <- printLine(s"Rating: ${review.rating}/10")
      _ <- printLine(s"Review: ${review.review}")
      _ <- printLine("Pros:")
      _ <- ZIO.foreach(review.pros)(pro => printLine(s"- $pro"))
      _ <- printLine("Cons:")
      _ <- ZIO.foreach(review.cons)(con => printLine(s"- $con"))
      _ <- printLine("")
    yield ()

  // Example 3: Chat-based structured output
  def chatBasedExample(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a JSON parser for MovieReview
    val parser = OutputParser.json[MovieReview]()
    
    // Chat messages
    val messages = Seq(
      ChatMessage.system("You are a helpful movie critic assistant."),
      ChatMessage.user("Can you review the movie 'Interstellar' (2014)?")
    )
    
    for
      _ <- printLine("Example 3: Chat-Based Structured Output")
      _ <- printLine("Generating structured movie review from chat...")
      review <- llm.completeChatStructured(messages, parser)
      _ <- printLine(s"Parsed review: ${review.toJson}")
      _ <- printLine(s"Movie: ${review.title} (${review.year})")
      _ <- printLine(s"Rating: ${review.rating}/10")
      _ <- printLine(s"Review: ${review.review}")
      _ <- printLine("Pros:")
      _ <- ZIO.foreach(review.pros)(pro => printLine(s"- $pro"))
      _ <- printLine("Cons:")
      _ <- ZIO.foreach(review.cons)(con => printLine(s"- $con"))
      _ <- printLine("")
    yield ()

  // Example 4: Using JSON schema validation
  def jsonSchemaExample(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a JSON schema for MovieReview
    val schema = JsonSchema.fromType[MovieReview]
    
    // Create a structured parser with schema validation
    val structuredParser = StructuredOutputParser.withJsonSchema[MovieReview](schema)
    
    // Prompt for the LLM
    val prompt = """
      |Write a detailed review of the movie "The Shawshank Redemption" (1994).
      |Include the title, year, your rating out of 10, a review, and lists of pros and cons.
      |""".stripMargin
    
    for
      _ <- printLine("Example 4: JSON Schema Validation")
      _ <- printLine("Generating structured movie review with schema validation...")
      review <- structuredParser.generateStructured(prompt, llm)
      _ <- printLine(s"Parsed review: ${review.toJson}")
      _ <- printLine(s"Movie: ${review.title} (${review.year})")
      _ <- printLine(s"Rating: ${review.rating}/10")
      _ <- printLine(s"Review: ${review.review}")
      _ <- printLine("Pros:")
      _ <- ZIO.foreach(review.pros)(pro => printLine(s"- $pro"))
      _ <- printLine("Cons:")
      _ <- ZIO.foreach(review.cons)(con => printLine(s"- $con"))
      _ <- printLine("")
    yield ()

  // Example 5: Handling parsing failures with retry
  def retryExample(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a JSON parser for a more complex structure
    case class ComplexReview(
      title: String,
      year: Int,
      director: String,
      cast: List[String],
      rating: Double,
      categories: List[String],
      review: String,
      analysis: Map[String, String]
    )
    
    implicit val complexReviewEncoder: JsonEncoder[ComplexReview] = DeriveJsonEncoder.gen[ComplexReview]
    implicit val complexReviewDecoder: JsonDecoder[ComplexReview] = DeriveJsonDecoder.gen[ComplexReview]
    
    val parser = OutputParser.json[ComplexReview]()
    
    // Prompt for the LLM
    val prompt = """
      |Write a detailed review of the movie "Pulp Fiction" (1994).
      |Include the title, year, director, main cast members, your rating out of 10,
      |categories (genres), a review, and an analysis with sections for cinematography,
      |screenplay, acting, and soundtrack.
      |""".stripMargin
    
    for
      _ <- printLine("Example 5: Retry Mechanism")
      _ <- printLine("Generating complex structured review with retry capability...")
      review <- parser.parseWithRetry(
        // Intentionally malformed JSON to trigger retry
        """
        |{
        |  "title": "Pulp Fiction",
        |  "year": 1994,
        |  "director": "Quentin Tarantino",
        |  "cast": ["John Travolta", "Samuel L. Jackson", "Uma Thurman"],
        |  "rating": 9.5,
        |  "categories": ["Crime", "Drama"],
        |  "review": "A groundbreaking film that redefined cinema in the 90s."
        |  // Missing the analysis field and has a syntax error (comment)
        |}
        |""".stripMargin,
        llm,
        maxRetries = 2
      ).catchAll { error =>
        printLine(s"Error: ${error.getMessage}") *>
        ZIO.succeed(ComplexReview(
          "Pulp Fiction", 1994, "Quentin Tarantino",
          List("John Travolta", "Samuel L. Jackson", "Uma Thurman"),
          9.5, List("Crime", "Drama"),
          "A groundbreaking film that redefined cinema in the 90s.",
          Map("note" -> "This is a fallback after parsing failure")
        ))
      }
      _ <- printLine(s"Parsed review: ${review.toJson}")
      _ <- printLine(s"Movie: ${review.title} (${review.year})")
      _ <- printLine(s"Director: ${review.director}")
      _ <- printLine(s"Cast: ${review.cast.mkString(", ")}")
      _ <- printLine(s"Rating: ${review.rating}/10")
      _ <- printLine(s"Categories: ${review.categories.mkString(", ")}")
      _ <- printLine(s"Review: ${review.review}")
      _ <- printLine("Analysis:")
      _ <- ZIO.foreach(review.analysis.toList) { case (key, value) =>
        printLine(s"- $key: $value")
      }
    yield ()

  // Main program
  override def run: ZIO[Any, Throwable, Unit] =
    // Create an OpenAI LLM
    val llm = for
      config <- ZIO.config(OpenAIConfig.config)
      llm <- OpenAILLM.createDefault(config)
    yield llm
    
    // Run all examples
    llm.flatMap { llmInstance =>
      for
        _ <- basicJsonParsingExample(llmInstance)
        _ <- llmExtensionExample(llmInstance)
        _ <- chatBasedExample(llmInstance)
        _ <- jsonSchemaExample(llmInstance)
        _ <- retryExample(llmInstance)
      yield ()
    }