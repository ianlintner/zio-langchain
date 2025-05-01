package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client
import zio.json.*

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.integrations.openai.*
import zio.langchain.parsers.*

/**
 * Example demonstrating structured output parsing with LLMs.
 *
 * This example shows how to use the parsers module to extract structured data from LLM responses.
 * It demonstrates both simple JSON parsing and schema-based parsing with validation.
 *
 * To run this example:
 * 1. Set your OpenAI API key in the environment variable OPENAI_API_KEY
 * 2. Run the example using: `sbt "examples/runMain zio.langchain.examples.StructuredOutputExample"`
 */
object StructuredOutputExample extends ZIOAppDefault:

  // Define some case classes for structured output
  case class Person(name: String, age: Int, occupation: String)
  object Person:
    implicit val encoder: JsonEncoder[Person] = DeriveJsonEncoder.gen[Person]
    implicit val decoder: JsonDecoder[Person] = DeriveJsonDecoder.gen[Person]

  case class MovieReview(title: String, year: Int, rating: Int, review: String)
  object MovieReview:
    implicit val encoder: JsonEncoder[MovieReview] = DeriveJsonEncoder.gen[MovieReview]
    implicit val decoder: JsonDecoder[MovieReview] = DeriveJsonDecoder.gen[MovieReview]

  // Define a JSON schema for movie reviews
  val movieReviewSchema = JsonSchema.fromString(
    """
    {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "description": "The title of the movie"
        },
        "year": {
          "type": "integer",
          "description": "The year the movie was released"
        },
        "rating": {
          "type": "integer",
          "description": "Rating from 1-10",
          "minimum": 1,
          "maximum": 10
        },
        "review": {
          "type": "string",
          "description": "A brief review of the movie"
        }
      },
      "required": ["title", "year", "rating", "review"]
    }
    """
  )

  val program = for
    // Print welcome message
    _ <- printLine("Welcome to the ZIO LangChain Structured Output Example!")
    _ <- printLine("")
    
    // Get the LLM service
    llm <- ZIO.service[LLM]
    
    // EXAMPLE 1: Simple structured output parsing
    _ <- printLine("=== EXAMPLE 1: Simple JSON Parsing ===")
    
    // Create a parser for the Person class
    personParser = StructuredOutputParser.forJson[Person]()
    
    // Define a prompt to generate a person's information
    prompt = "Generate information about a fictional person."
    
    // Use the parser with the LLM
    _ <- printLine("Generating a structured person object...")
    person <- personParser.generateStructured(prompt, llm)
    
    // Print the result
    _ <- printLine(s"Generated person: ${person.toJson}")
    _ <- printLine(s"Name: ${person.name}")
    _ <- printLine(s"Age: ${person.age}")
    _ <- printLine(s"Occupation: ${person.occupation}")
    _ <- printLine("")
    
    // EXAMPLE 2: Schema-based parsing with validation
    _ <- printLine("=== EXAMPLE 2: Schema-based Parsing ===")
    
    // Create a parser with the movie review schema
    movieParser = JsonSchemaOutputParser.apply[MovieReview](movieReviewSchema)
    structuredParser = StructuredOutputParser(movieParser)
    
    // Define a prompt to generate a movie review
    moviePrompt = "Generate a review for a science fiction movie."
    
    // Use the parser with the LLM
    _ <- printLine("Generating a structured movie review...")
    review <- structuredParser.generateStructured(moviePrompt, llm)
    
    // Print the result
    _ <- printLine(s"Generated review: ${review.toJson}")
    _ <- printLine(s"Title: ${review.title}")
    _ <- printLine(s"Year: ${review.year}")
    _ <- printLine(s"Rating: ${review.rating}/10")
    _ <- printLine(s"Review: ${review.review}")
    _ <- printLine("")
    
    // EXAMPLE 3: Chat-based structured output parsing
    _ <- printLine("=== EXAMPLE 3: Chat-based Structured Output Parsing ===")
    
    // Define a chat prompt to generate a person's information in a chat context
    chatMessages = Seq(
      ChatMessage.system("You are a helpful assistant that generates structured information."),
      ChatMessage.user("Generate information about a fictional musician.")
    )
    
    // Create a parser for the Person class to use with chat
    musicianParser = StructuredOutputParser.forJson[Person]()
    
    // Use the parser's generateStructuredChat method
    _ <- printLine("Generating a structured person from chat...")
    chatPerson <- musicianParser.generateStructuredChat(chatMessages, llm)
    
    // Print the result
    _ <- printLine(s"Generated musician: ${chatPerson.toJson}")
    _ <- printLine(s"Name: ${chatPerson.name}")
    _ <- printLine(s"Age: ${chatPerson.age}")
    _ <- printLine(s"Occupation: ${chatPerson.occupation}")
    _ <- printLine("")
    
    // Final message
    _ <- printLine("All examples completed successfully!")
  yield ()

  override def run = program.provide(
    // HTTP Client dependency required by OpenAI integration
    Client.default,
    
    // OpenAI LLM layer
    OpenAILLM.live,
    
    // OpenAI configuration layer
    ZLayer.succeed(
      OpenAIConfig(
        apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
        model = "gpt-3.5-turbo"
      )
    )
  )